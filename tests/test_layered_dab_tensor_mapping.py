import numpy as np
import torch as t
from mcts_playground.games.dots_and_boxes import DotsAndBoxesState, CellType
from experiments.dots_and_boxes_experiments.encoder import LayeredDABTensorMapping

def test_encode_states_shape():
    """The tensor should have shape (batch, 3, H, W)."""
    state = DotsAndBoxesState(rows=1, cols=1)
    device = t.device("cpu")
    encoded = LayeredDABTensorMapping.encode_states([state], device)
    assert encoded.shape == (1, 3, 2 * state.rows + 1, 2 * state.cols + 1)


def test_edges_channel():
    """Taken edges should be marked with 1 in channel 0."""
    state = DotsAndBoxesState(rows=1, cols=1)
    # Draw a single horizontal edge on the top of the square
    state.apply_action((0, 1))  # (row 0, col 1) is a horizontal edge position
    device = t.device("cpu")
    encoded = LayeredDABTensorMapping.encode_states([state], device)[0]

    # Build expected edge mask
    board = state.board
    expected_edges = np.logical_or(board == CellType.VERTICAL_EDGE, board == CellType.HORIZONTAL_EDGE).astype(np.float32)
    encoded_edges = encoded[0].cpu().numpy()

    assert np.array_equal(encoded_edges, expected_edges), "Edge channel does not match expected mask"


def test_box_channels_after_capture():
    """After completing a square, the corresponding player's box channel should be 1 at that square."""
    state = DotsAndBoxesState(rows=1, cols=1)
    # Sequence of moves to complete the single square
    # A draws top edge
    state.apply_action((0, 1))
    # B draws left edge
    state.apply_action((1, 0))
    # A draws bottom edge
    state.apply_action((2, 1))
    # B draws right edge – completes the square for B, they get the point
    state.apply_action((1, 2))

    device = t.device("cpu")
    encoded = LayeredDABTensorMapping.encode_states([state], device)[0]

    # The completed square is at board position (1, 1)
    completed_square_row, completed_square_col = 1, 1

    current_player_box_channel = encoded[1].cpu().numpy()
    other_player_box_channel = encoded[2].cpu().numpy()

    # Since B captured the square, but A will be current player switch logic, ensure exactly one of the channels is 1.
    assert (current_player_box_channel[completed_square_row, completed_square_col] == 1) ^ (
        other_player_box_channel[completed_square_row, completed_square_col] == 1
    ), "Exactly one player should own the completed square" 