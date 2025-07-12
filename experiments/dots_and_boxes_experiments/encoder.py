import torch as t
import numpy as np
from mcts_playground.games.dots_and_boxes import DotsAndBoxesState, DotsAndBoxesAction, DotsAndBoxesPlayer, CellType
import torch.nn.functional as F
from typing import Dict, Tuple, Any, List
from mcts_playground import TensorMapping
from mcts_playground.algorithms.AlphaZero import AlphaZeroEvaluation
from mcts_playground.data_structures import TrainingExample

# Functions to convert between edge coordinates and flat indices.
# The edges are enumerated by first listing all vertical edges (left to right, top to bottom),
# then all horizontal edges (top to bottom, left to right).
def edge_to_idx(row: int, col: int, num_rows: int, num_cols: int) -> int:
    if col % 2 == 0:  # vertical edge
        idx = row // 2 * (num_cols + 1) + col // 2
    else:  # horizontal edge
        idx = col // 2 * (num_rows + 1) + row // 2
        idx += num_rows * (num_cols + 1) 
    return idx

def idx_to_edge(idx: int, num_rows: int, num_cols: int) -> Tuple[int, int]:
    if idx < num_rows * (num_cols + 1):  # vertical edge
        row = idx // (num_cols + 1)
        row = 2 * row + 1 
        col = idx % (num_cols + 1)
        col = 2 * col
    else:  # horizontal edge
        idx_res = idx - num_rows * (num_cols + 1) 
        row = idx_res % (num_rows + 1)
        row = 2 * row 
        col = idx_res // (num_rows + 1)
        col = 2 * col + 1
    return (row, col)

class DABTensorMapping(TensorMapping[DotsAndBoxesAction, AlphaZeroEvaluation[DotsAndBoxesAction, DotsAndBoxesPlayer]]):
    """
    Base class for Dots and Boxes model interfaces containing common functionality.
    Edges (i,j) to indices idx enumeration via first listing all vertical edges, then horizontal ones; left to right, top to bottom.
    """
    @staticmethod
    def decode_outputs(outputs: Dict[str, t.Tensor], states: List[DotsAndBoxesState]) -> List[AlphaZeroEvaluation[DotsAndBoxesAction, DotsAndBoxesPlayer]]:
        """Convert raw model output to policies and values."""
        policy_logits = outputs["policy"]
        value = outputs["value"]

        num_rows, num_cols = states[0].rows, states[0].cols
        legal_actions_list = [state.legal_actions for state in states]

        # Create a mask for the legal actions
        masks = t.full_like(policy_logits, float('-inf'))
        for i, legal_actions in enumerate(legal_actions_list):
            for (row, col) in legal_actions:
                idx = edge_to_idx(row, col, num_rows, num_cols)
                masks[i][idx] = 0.0

        # Apply mask and convert to probabilities
        masked_logits = policy_logits + masks
        policy_probs = F.softmax(masked_logits, dim=-1)

        # Convert to action->probability dictionary
        policy_dict = [{} for _ in states]
        for i, legal_actions in enumerate(legal_actions_list):
            for idx in range(2 * num_rows * num_cols + num_rows + num_cols):  # number of edges in the game
                prob = policy_probs[i][idx].item()  # Convert to float
                row, col = idx_to_edge(idx, num_rows, num_cols)
                if (row, col) in legal_actions:
                    policy_dict[i][(row, col)] = prob

        # Convert value for current player into player->value dictionary
        value_dict = [{} for _ in states]
        for i, value in enumerate(value):
            match states[i].current_player:
                case 'A':
                    value_dict[i] = {'A': value.item(), 'B': -value.item()}
                case 'B':
                    value_dict[i] = {'A': -value.item(), 'B': value.item()}
                case _:
                    raise ValueError(f"Invalid player: {states[i].current_player}")
        
        # Validation assertions
        assert all(set(policy_dict[i].keys()) == set(legal_actions_list[i]) for i in range(len(states))), "Policy dictionary keys do not match legal actions"
        assert all(abs(sum(policy_dict[i].values()) - 1.0) < 1e-2 for i in range(len(states))), "Policy dictionary values do not sum to 1"
        
        return list(zip(policy_dict, value_dict))
    
    @staticmethod
    def encode_targets(examples: List[TrainingExample[DotsAndBoxesAction, AlphaZeroEvaluation]], device: t.device) -> Tuple[Dict[str, t.Tensor], Dict[str, t.Tensor]]:
        """Convert a training example into tensor targets and auxiliary data for loss computation.
        
        Args:
            examples: List of TrainingExamples containing targets of (policy_dict, value)
        
        Returns:
            Tuple of (targets, data), where targets is a dictionary of tensors and 
            data is a dictionary of auxiliary tensors
        """
        policy_dict_list, value_list = zip(*[example.target for example in examples])
        num_rows, num_cols = examples[0].state.rows, examples[0].state.cols # type: ignore
        N_edges = 2 * num_rows * num_cols + num_rows + num_cols
        
        # Convert policy dict to tensor
        policy = t.zeros(len(examples), N_edges, device=device, dtype=t.float32)
        for i, policy_dict in enumerate(policy_dict_list):
            for (row, col), prob in policy_dict.items():
                idx = edge_to_idx(row, col, num_rows, num_cols)
                policy[i][idx] = prob
        
        # Convert legal actions to a tensor mask if available
        data = {}
        if examples and hasattr(examples[0], 'extra_data') and "legal_actions" in examples[0].extra_data:
            legal_actions_list = [example.extra_data["legal_actions"] for example in examples]
            legal_actions = t.empty(len(examples), N_edges, device=device, dtype=t.bool)
            for i, legal_actions_per_example in enumerate(legal_actions_list):
                for row, col in legal_actions_per_example:
                    idx = edge_to_idx(row, col, num_rows, num_cols)
                    legal_actions[i][idx] = True
            data["legal_actions"] = legal_actions
        
        return {
            "policy": policy,
            "value": t.tensor(value_list, device=device, dtype=t.float32)
        }, data
    
    @staticmethod
    def encode_states(states: List[DotsAndBoxesState], device: t.device) -> t.Tensor:
        """Convert board state to neural network input tensor.
        
        Input: game states with boards of size (2*rows+1) x (2*cols+1)
        Output: PyTorch tensor encoding of shape (batch_size, 4 * num_rows * num_cols + num_rows + num_cols). 
            Concatenation of the following:
            - vertical edges (0 if not taken, 1 if taken), total num_rows*(num_cols + 1) features
            - horizontal edges (0 if not taken, 1 if taken), total (num_rows + 1) * num_cols features
            - boxes owned by current player (0 if not owned, 1 if owned), total num_rows * num_cols features
            - boxes owned by other player (0 if not owned, 1 if owned), total num_rows * num_cols features
        """

        if not states:
            raise ValueError("Empty states list")
        
        # Initialize useful variables
        batch_size = len(states)
        boards = np.stack([state.board for state in states])  # (batch_size, 2*num_rows + 1, 2*num_cols + 1)
        current_player_cell_types = np.array([CellType.PLAYER_A_SQUARE if state.current_player == 'A' else CellType.PLAYER_B_SQUARE for state in states]).reshape(batch_size, 1, 1) # (batch_size, 1, 1)
        other_player_cell_types = np.array([CellType.PLAYER_B_SQUARE if state.current_player == 'A' else CellType.PLAYER_A_SQUARE for state in states]).reshape(batch_size, 1, 1) # (batch_size, 1, 1)
        
        # Process vertical edges
        vertical_edges = boards[:, 1::2, ::2] # (batch_size, num_rows, num_cols + 1)
        vertical_edges_mask = np.where(vertical_edges == CellType.VERTICAL_EDGE, 1, 0).reshape(batch_size, -1) # (batch_size, num_rows*(num_cols + 1))

        # Process horizontal edges
        horizontal_edges = boards[:, ::2, 1::2] # (batch_size, num_rows + 1, num_cols)
        horizontal_edges_mask = np.where(horizontal_edges == CellType.HORIZONTAL_EDGE, 1, 0).transpose(0, 2, 1).reshape(batch_size, -1) # (batch_size, (num_rows + 1) * num_cols)

        # Process boxes
        boxes = boards[:, 1::2, 1::2] # (batch_size, num_rows, num_cols)
        current_player_mask = np.where(boxes == current_player_cell_types, 1, 0).reshape(batch_size, -1) # (batch_size, num_rows*num_cols)
        other_player_mask = np.where(boxes == other_player_cell_types, 1, 0).reshape(batch_size, -1) # (batch_size, num_rows*num_cols)

        res = np.concatenate((
            vertical_edges_mask,
            horizontal_edges_mask,
            current_player_mask,
            other_player_mask
        ), axis=1)
        res = t.from_numpy(res).to(device)
        return res
    
class LayeredDABTensorMapping(DABTensorMapping):
    @staticmethod
    def encode_states(states: List[DotsAndBoxesState], device: t.device) -> t.Tensor:
        """Convert board state to neural network input tensor.
        
        Input: game states with boards of size (2*rows+1) x (2*cols+1)
        Output: PyTorch tensor encoding of shape (batch_size, 3, 2 * num_rows + 1, 2 * num_cols + 1). 
            - Channel 0: 1 for positions corresponding to taken edges, 0 otherwise.
            - Channel 1: 1 for boxes captured by the current player, 0 otherwise.
            - Channel 2: 1 for boxes captured by the other player, 0 otherwise.
        """
        if not states:
            raise ValueError("Empty states list")
        
        num_rows, num_cols = states[0].rows, states[0].cols
        
        boards = np.array([state.board for state in states]) # (num_states, 2*num_rows + 1, 2*num_cols + 1)
        result = np.zeros((len(states), 3, 2 * num_rows + 1, 2 * num_cols + 1), dtype=np.float32) # initialize output array

        # Process edges
        result[:, 0, :, :] = np.logical_or(boards == CellType.VERTICAL_EDGE, boards == CellType.HORIZONTAL_EDGE).astype(np.float32)
        
        # Process current player boxes
        current_player_boxes = np.array([CellType.PLAYER_A_SQUARE if state.current_player == 'A' else CellType.PLAYER_B_SQUARE for state in states]).reshape(len(states), 1, 1)
        result[:, 1, :, :] = np.where(boards == current_player_boxes, 1.0, 0.0)

        # Process other player boxes
        other_player_boxes = np.array([CellType.PLAYER_B_SQUARE if state.current_player == 'A' else CellType.PLAYER_A_SQUARE for state in states]).reshape(len(states), 1, 1)
        result[:, 2, :, :] = np.where(boards == other_player_boxes, 1.0, 0.0)

        # Convert to float32 tensor on the requested device
        res = t.from_numpy(result).to(device=device, dtype=t.float32)
        return res
    

# class DABTokenizedTensorMapping(DABBaseTensorMapping):
#     @staticmethod
#     def encode_states(states: List[DotsAndBoxesState], device: t.device) -> t.Tensor:
#         """Convert board state to neural network input tensor.

#         Input: Game states with boards of size (2*rows+1) x (2*cols+1)

#         Output: PyTorch tensor encoding:
#         - vertical edges (0 if not taken, 1 if taken),
#         - horizontal edges (2 if not taken, 3 if taken),
#         - box ownership (4 if not taken, 5 if owned by A, 6 if owned by B). 
#         """
#         boards = np.array([state.board for state in states]) # (num_states, 2*num_rows + 1, 2*num_cols + 1)
#         #Process vertical edges
#         VerticalEdges = boards[:, 1::2, ::2] # (num_states, num_rows, num_cols + 1)
#         res_top = np.where(VerticalEdges == CellType.VERTICAL_EDGE, 1, 0).reshape(len(states), -1) # (num_states, num_rows*(num_cols + 1))
#         #Process horizontal edges
#         HorizontalEdges = boards[:, ::2, 1::2] # (num_states, num_rows + 1, num_cols)
#         res_bottom = np.transpose(np.where(HorizontalEdges == CellType.HORIZONTAL_EDGE, 3, 2), axes=(0,2,1)).reshape(len(states), -1) # (num_states, (num_rows + 1) * num_cols)
#         #Process boxes
#         Boxes = boards[:, 1::2, 1::2] # (num_states, num_rows, num_cols)
#         res_boxes = np.where(Boxes == CellType.PLAYER_A_SQUARE, 5, np.where(Boxes == CellType.PLAYER_B_SQUARE, 6, 4)).reshape(len(states), -1) # (num_states, num_rows*num_cols)
#         #Concatenate
#         result = np.concatenate((res_top, res_bottom, res_boxes), axis = 1) # (num_states, num_rows*(num_cols + 1) + (num_rows + 1) * num_cols + num_rows*num_cols)
#         encoded_board = t.from_numpy(result)
#         encoded_board = encoded_board.to(device=device)
#         return encoded_board