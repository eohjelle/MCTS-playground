from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import math
import random

# A (game) state encodes a node in the entire game tree
class State(ABC):
    def __init__(self, terminal: bool, value: Optional[float] = None):
        self.terminal = terminal
        self.value = value

    @abstractmethod
    def childStates(self) -> List['State']:
        pass

# A predictor predicts the policy and value of a state. 
# Examples: Deep learning model, Monte Carlo predictor
class Predictor(ABC):
    @abstractmethod
    def predict(self, state: State) -> Tuple[float, float]:
        # Returns (policy, value)
        pass

# Node is a node in the search tree. 
class Node:
    def __init__(self, parent: Optional['Node'], children: List['Node'], value: int, policy: Optional[Dict['Node', float]], state: State):
        self.parent = parent
        self.children = children
        self.value = value
        self.policy = policy
        self.state = state
        self.visitCount = 0

    def isLeaf(self) -> bool:
        return len(self.children) == 0

class TreeSearchAlgorithm:
    def __init__(self, root: Node, predictor: Predictor):
        self.root = root
        self.predictor = predictor
    
    # Run the tree search algorithm numIterations times and return an action
    # TODO: Return the policy, not just the action
    def run(self, numIterations: int, temperature: float = 1.0):
        for _ in range(numIterations):
            self.treeSearchIteration()
        
        # Use temperature to sample from visit counts: sample proportional to child.visitCount^(1/temperature)
        if temperature == 0:
            # If temperature is 0, select greedily
            return max(self.root.children, key=lambda child: child.visitCount)
        else:
            weights = [child.visitCount ** (1/temperature) for child in self.root.children]
            return random.choices(self.root.children, weights=weights, k=1)[0]

    def treeSearchIteration(self):
        leaf = self.select()
        if not leaf.state.terminal:
            self.expand(leaf)
        self.evaluate(leaf)
        self.backpropagate(leaf)

    def select(self) -> Node:
        # TODO: Use better UCT formula
        def UCT(node: Node) -> float:
            return node.value + node.parent.policy[node] * math.sqrt(math.log(node.parent.visitCount + 1) / (node.visitCount + 1))

        node = self.root
        while not node.isLeaf():
            node = max(node.children, key=UCT)
        return node
    
    def expand(self, node: Node):
        if not node.state.terminal:
            for childState in node.state.childStates():
                node.children.append(Node(node, [], 0, None, childState))

    def evaluate(self, node: Node):
        if node.state.terminal:
            node.value = node.state.value
        else:
            node.policy, node.value = self.predictor.predict(node.state)

    def backpropagate(self, node: Node, value: float):
        while node is not None:
            node.visitCount += 1
            node.value = ((node.visitCount - 1) * node.value + value) / node.visitCount
            node = node.parent