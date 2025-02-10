from typing import Tuple, TypeVar, Optional, Protocol, Generic, List, Dict

# Define some types
ActionType = TypeVar('ActionType')  # type of actions at a state in the game
ValueType = TypeVar('ValueType')    # type of value stored in the node, e.g. policy and expected reward
OutcomeType = TypeVar('OutcomeType')  # type of optional outcome from evaluation
DataType = TypeVar('DataType')      # type of optional data from evaluation

class State(Protocol[ActionType]):
    def get_legal_actions(self) -> List[ActionType]:
        pass
    
    def is_terminal(self) -> bool:
        pass

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

    def expand(self) -> None:
        for action in self.state.get_legal_actions():
            new_state = self.state.apply_action(action)
            child = Node(new_state, parent=self)
            self.children[action] = child

class TreeSearch(Protocol[ActionType, ValueType, OutcomeType, DataType]):
    """Protocol for tree search algorithms like MCTS."""
    root: Node[ActionType, ValueType]
    
    def select(self, node: Node[ActionType, ValueType]) -> ActionType:
        """Select an action at the given node during tree traversal."""
        pass
    
    def evaluate(self, state: State[ActionType]) -> Tuple[ValueType, Optional[OutcomeType], Optional[DataType]]:
        """Evaluate a leaf node's state."""
        pass
    
    def update(self, node: Node[ActionType, ValueType], action: Optional[ActionType], value: ValueType, outcome: Optional[OutcomeType]) -> None:
        """Update a node's value during backpropagation."""
        pass
    
    def policy(self, node: Node[ActionType, ValueType]) -> ActionType:
        """Select the best action at a node according to the search results."""
        pass
    
    def __call__(self, num_simulations: int) -> Tuple[ActionType, List[DataType]]:
        """Run simulations and return the best action according to the policy.
        
        This is a concrete implementation that uses the abstract methods above.
        """
        collected_data: List[DataType] = []

        if self.root.is_leaf():
            self.root.expand()

        for _ in range(num_simulations):
            node = self.root
            path = []  # List of (node, action) pairs for backpropagation

            # Selection
            while not node.is_leaf() and not node.state.is_terminal():
                action = self.select(node)
                path.append((node, action))
                node = node.children[action]
            
            path.append((node, None))  # Leaf node action is None

            # Evaluation
            value, outcome, data = self.evaluate(node.state)

            # Expansion
            if not node.state.is_terminal():
                node.expand()

            # Backpropagation
            for node, action in reversed(path):
                self.update(node, action, value, outcome)
            
            # Collect data
            if data is not None:
                collected_data.append(data)
        
        return self.policy(self.root), collected_data
    
    def update_root(self, actions: List[ActionType]) -> None:
        """Update the root node after committing to actions."""
        for action in actions:
            self.root = self.root.children[action]