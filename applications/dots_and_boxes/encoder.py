from typing import Tuple
import numpy as np
import torch as t
from game_state import *


def simpleEncode(GameState: DotsAndBoxesGameState) -> t.Tensor: 
    '''
    Input: game state with board of size MAX_SIZE x MAX_SIZE; hence having 2*MAX_SIZE*(MAX_SIZE+1) edges
    Output: pyTorch matrix of size (2*MAX_SIZE)x(MAX_SIZE+1); the first MAX_SIZE rows encode placed VERTICAL edges
    and the last MAX_SIZE rows encode the transpose of the HORIZONTAL edges
    '''
    board = GameState.board
    #Process VERTICALS
    VerticalEdges = board[1::2, ::2]
    res_top = np.where(VerticalEdges == VERTICAL, 1, 0)
    #Process HORIZONTALS
    HorizontalEdges = board[::2, 1::2]
    res_bottom = np.transpose(np.where(HorizontalEdges == HORIZONTAL, 1, 0))
    #Concatenate
    result = np.concatenate((res_top, res_bottom), axis = 0)
    return t.from_numpy(result)

def simpleDecode(x: t.Tensor) -> Tuple[int,int]:
    ''' 
    Input: (2*MAX_SIZE)x(MAX_SIZE+1) tensor with a single 1
    Output: (i,j) corresponding to the edge encoded by the tensor
    '''
    #TODO
    pass


def multiLayerEncode(GameState: DotsAndBoxesGameState) -> t.Tensor:
    '''
    Input: game state with board of size MAX_SIZE x MAX_SIZE; hence having 2*MAX_SIZE*(MAX_SIZE+1) edges
    Output: pyTorch tensor of size 3x(2*MAX_SIZE+1)x(2*MAX_SIZE+1); the first layer has +/-1s according to who owns an edge, else zero;
    second layer has a +/-1 according to who owns a box, else zero; third layer is constant +/-1 encoding whose turn it is
    '''
    board = GameState.board
    N=2*MAX_SIZE+1
    #first layer
    firstLayer = np.zeros((N,N))
    edge_dict = GameState.edge_owners
    if len(edge_dict) !=0:
        indices = np.array(list(edge_dict.keys()))
        values = np.array(list(edge_dict.values()))
        firstLayer[tuple(indices.T)] = values
    
    #second layer
    secondLayer = np.where(board==PLAYER_SYMBOLS[P1], 1, np.where(board==PLAYER_SYMBOLS[P2],-1, 0))

    #third layer
    thirdLayer = np.ones((N,N))*GameState.player_turn

    result = np.stack((firstLayer, secondLayer, thirdLayer), axis = 0)
    return result

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