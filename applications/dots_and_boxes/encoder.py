import numpy as np
import torch as t
from applications.dots_and_boxes.game_state import *
import torch.nn.functional as F
from typing import Dict, Tuple, Any
from core import TensorMapping
from core.implementations.AlphaZero import AlphaZeroTarget
from core.data_structures import TrainingExample

class DABBaseTensorMapping(TensorMapping[DotsAndBoxesAction, AlphaZeroTarget]):
    """
    Base class for Dots and Boxes model interfaces containing common functionality.
    Edges (i,j) to indices idx enumeration via first listing all vertical edges, then horizontal ones; laft to right, top to bottom.
    """
    @staticmethod
    def decode_output(output: Dict[str, t.Tensor], state: DotsAndBoxesGameState) -> AlphaZeroTarget:
        """Convert raw model output to policy dictionary and value."""
        policy_logits = output["policy"]
        value = output["value"]

        policy_probs = F.softmax(policy_logits, dim=0)

        num_rows, num_cols = state.rows, state.cols

        
        # Convert to action->probability dictionary
        policy_dict = {}
        legal_actions = state.get_legal_actions()
        for idx in range(2*num_rows*num_cols+num_rows+num_cols):  #number of edges in the game
            prob = policy_probs[idx].item()  # Convert to float
            # Convert flat index back to (row, col)
            if idx < num_rows*(num_cols+1): # vertical edge
                row = idx // (num_cols+1)
                row = 2*row+1 
                col = idx % (num_cols+1)
                col = 2*col
            else: # horizontal edge
                idx_res = idx - num_rows*(num_cols+1) 
                row = idx_res % (num_rows+1)
                row = 2*row 
                col = idx_res // (num_rows+1)
                col = 2*col+1
            if (row, col) in legal_actions:
                policy_dict[(row, col)] = prob
        assert set(policy_dict.keys()) == set(legal_actions), "Policy dictionary keys do not match legal actions"
        return policy_dict, float(value.item())  # Convert value to float
    
    @staticmethod
    def encode_example(example: TrainingExample[DotsAndBoxesAction, AlphaZeroTarget], device: t.device) -> Tuple[Dict[str, t.Tensor], Dict[str, Any]]:
        """Convert a training example into tensor targets and auxiliary data for loss computation.
        
        Args:
            example: TrainingExample containing a target of (policy_dict, value)
        
        Returns:
            Tuple of (targets, data), where targets is a dictionary of tensors and 
            data is a dictionary of auxiliary tensors
        """
        policy_dict, value = example.target

        num_rows, num_cols = example.state.rows, example.state.cols # type: ignore
        
        # Convert policy dict to tensor
        policy = t.zeros(2*num_rows*num_cols+num_rows+num_cols, device=device)
        for (row, col), prob in policy_dict.items():
            if col%2==0: # vertical edge
                idx = row//2 * (num_cols+1) + col//2
            else: # horizontal edge
                idx = col//2 * (num_rows+1) + row//2
                idx += num_rows*(num_cols+1)  
            policy[idx] = prob
        
        # Convert legal actions to a tensor mask if available
        data = {}
        if "legal_actions" in example.data:
            legal_actions_list = example.data["legal_actions"]
            legal_actions = t.zeros(2*num_rows*num_cols+num_rows+num_cols, device=device)
            for action in legal_actions_list:
                row, col = action
                if col%2==0: # vertical edge
                    idx = row//2 * (num_cols+1) + col//2 
                else: # horizontal edge
                    idx = col//2 * (num_rows+1) + row//2
                    idx += num_rows*(num_cols+1)
                legal_actions[idx] = 1.0
            data["legal_actions"] = legal_actions
        
        return {
            "policy": policy,
            "value": t.tensor(value, device=device, dtype=t.float32)
        }, data
    
class DABSimpleTensorMapping(DABBaseTensorMapping):
    @staticmethod
    def encode_state(state: DotsAndBoxesGameState, device: t.device) -> t.Tensor:
        """Convert board state to neural network input tensor.
        Input: game state with board of size MAX_SIZE x MAX_SIZE; hence having 2*MAX_SIZE*(MAX_SIZE+1) edges
        Output: pyTorch matrix of size (2*MAX_SIZE)x(MAX_SIZE+1); the first MAX_SIZE rows encode placed VERTICAL edges
        and the last MAX_SIZE rows encode the transpose of the HORIZONTAL edges
        """
        board = state.board
        #Process VERTICALS
        VerticalEdges = board[1::2, ::2]
        res_top = np.where(VerticalEdges == VERTICAL, 1, 0).flatten()
        #Process HORIZONTALS
        HorizontalEdges = board[::2, 1::2]
        res_bottom = np.transpose(np.where(HorizontalEdges == HORIZONTAL, 1, 0)).flatten()
        #Concatenate
        result = np.concatenate((res_top, res_bottom), axis = 0)
        encoded_board = t.from_numpy(result)
        encoded_board = encoded_board.to(device=device)
        return encoded_board

class DABMultiLayerTensorMapping(DABBaseTensorMapping):
    @staticmethod
    def encode_state(state: DotsAndBoxesGameState, device: t.device) -> t.Tensor:
        """Convert board state to neural network input tensor.
        
        Input: game state with board of size MAX_SIZE x MAX_SIZE; hence having 2*MAX_SIZE*(MAX_SIZE+1) edges
        Output: pyTorch tensor of size 3x(2*MAX_SIZE+1)x(2*MAX_SIZE+1); the first layer has +/-1s according to who owns an edge, else zero;
        second layer has a +/-1 according to who owns a box, else zero; third layer is constant +/-1 encoding whose turn it is
        """
        board = state.board
        N=2*MAX_SIZE+1
        #first layer
        firstLayer = np.zeros((N,N))
        edge_dict = state.edge_owners
        if len(edge_dict) !=0:
            indices = np.array(list(edge_dict.keys()))
            values = np.array(list(edge_dict.values()))
            firstLayer[tuple(indices.T)] = values
        
        #second layer
        secondLayer = np.where(board==PLAYER_SYMBOLS[P1], 1, np.where(board==PLAYER_SYMBOLS[P2],-1, 0))

        #third layer
        thirdLayer = np.ones((N,N))*state.player_turn

        encoded_board = np.stack((firstLayer, secondLayer, thirdLayer), axis = 0)
        encoded_board = t.from_numpy(encoded_board)
        encoded_board = encoded_board.to(device=device)
        return encoded_board.flatten()


'''
#Test
x = GameState()
x.play(1,2)
x.play(2,1)
x.play(0,1)
x.play(1,0)
x.play(0,5)
print("SimpleEncode:", simpleEncode(x))
print("MultiLayerEncode:", multiLayerEncode(x))
'''