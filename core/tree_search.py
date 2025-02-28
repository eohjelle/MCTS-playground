from typing import Optional, Generic, List, Dict
from abc import ABC, abstractmethod
from core.types import ActionType, ValueType, EvaluationType
from core.state import State

class Node(Generic[ActionType, ValueType]):
    def __init__(
        self, 
        state: State[ActionType],
        parent: Optional['Node[ActionType, ValueType]'] = None,
        value: Optional[ValueType] = None
    ):
        self.state = state
        self.parent = parent
        self.children: Dict[ActionType, Node[ActionType, ValueType]] = {}
        self.value = value

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def expand(self, actions: Optional[List[ActionType]] = None) -> None:
        if actions is None:
            actions = self.state.get_legal_actions()
        for action in actions:
            if action not in self.children:
                new_state = self.state.apply_action(action)
                child = Node(new_state, parent=self)
                self.children[action] = child
        

class TreeSearch(ABC, Generic[ActionType, ValueType, EvaluationType]):
    """Protocol for tree search algorithms like MCTS."""
    root: Node[ActionType, ValueType]
    num_simulations: int
    
    @abstractmethod
    def select(self, node: Node[ActionType, ValueType]) -> ActionType:
        """Select an action at the given node during tree traversal."""
        pass
    
    @abstractmethod
    def evaluate(self, node: Node[ActionType, ValueType]) -> EvaluationType:
        """Evaluate a leaf node's state."""
        pass
    
    @abstractmethod
    def update(self, node: Node[ActionType, ValueType], action: Optional[ActionType], evaluation: EvaluationType) -> None:
        """Update a node's value during backpropagation."""
        pass
    
    @abstractmethod
    def policy(self, node: Node[ActionType, ValueType]) -> ActionType:
        """Select the best action at a node according to the search results."""
        pass
    
    def __call__(self) -> ActionType:
        """Run simulations and return the best action according to the policy.
        This is a concrete implementation that uses the abstract methods above.
        """

        # If the root node is a leaf node, it is initialized as if it has been visited before during tree search
        if self.root.is_leaf():
            self.root.expand()
            evaluation = self.evaluate(self.root)
            self.update(self.root, None, evaluation)

        for _ in range(self.num_simulations):
            node = self.root
            path = []  # List of (node, action) pairs for backpropagation

            # Selection
            while not node.is_leaf() and not node.state.is_terminal():
                action = self.select(node)
                path.append((node, action))
                node = node.children[action]
            
            path.append((node, None))  # Leaf node action is None

            # Expansion
            if not node.state.is_terminal():
                node.expand()

            # Evaluation
            evaluation = self.evaluate(node)

            # Backpropagation, including the leaf node
            for node, action in reversed(path):
                self.update(node, action, evaluation)
        
        return self.policy(self.root)
    
    def update_root(self, actions: List[ActionType]) -> None:
        """Update the root node after committing to actions."""
        for action in actions:
            self.root.expand([action]) # May have to change this or handle unexpanded node if we want to reanalyze
            self.root = self.root.children[action]
