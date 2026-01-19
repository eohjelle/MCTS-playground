from typing import Optional, Generic, List, Dict, Protocol, Self
from dataclasses import dataclass, field
from mcts_playground.types import ActionType, ValueType, PlayerType, EvaluationType
from mcts_playground.state import State

# TODO: Implement a serialize method of State and store that in Node etc instead of whole State classes
# TODO: Make Node into Protocol. The "value" attribute becomes redundant and part of what any implementation decides to store.
# TODO: Make expand part of the TreeSearch class (perhaps with a default implementation).
# TODO: Change update(node, action, evaluation) to backpropagate(path, evaluation).
# TODO (Low priority): Use yield pattern for parallel MCTS


@dataclass
class Node(Generic[ActionType, ValueType, PlayerType]):
    state: State[ActionType, PlayerType]
    value: Optional[ValueType] = None
    children: Dict[ActionType, "Node[ActionType, ValueType, PlayerType]"] = field(
        default_factory=dict
    )

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def expand(
        self,
        state_dict: Dict[
            State[ActionType, PlayerType], "Node[ActionType, ValueType, PlayerType]"
        ],
        actions: Optional[List[ActionType]] = None,
    ) -> None:
        if actions is None:
            actions = self.state.legal_actions
        for action in actions:
            if action not in self.children:
                new_state = self.state.clone()
                new_state.apply_action(action)
                if new_state in state_dict:
                    child = state_dict[new_state]
                else:
                    child = Node[ActionType, ValueType, PlayerType](new_state)
                    state_dict[new_state] = child
                self.children[action] = child


class TreeSearch(Protocol[ActionType, ValueType, EvaluationType, PlayerType]):
    """Protocol for tree search algorithms like MCTS."""

    root: Node[ActionType, ValueType, PlayerType]
    num_simulations: int
    state_dict: Dict[
        State[ActionType, PlayerType], Node[ActionType, ValueType, PlayerType]
    ]

    def select(self, node: Node[ActionType, ValueType, PlayerType]) -> ActionType:
        """Select an action at the given node during tree traversal."""
        ...

    def evaluate(self, node: Node[ActionType, ValueType, PlayerType]) -> EvaluationType:
        """Evaluate a leaf node's state."""
        ...

    def update(
        self,
        node: Node[ActionType, ValueType, PlayerType],
        action: Optional[ActionType],
        evaluation: EvaluationType,
    ) -> EvaluationType:
        """Update a node's value during backpropagation."""
        ...

    def policy(self) -> ActionType:
        """Select the best action at the root node according to the search results."""
        ...

    def __call__(self) -> ActionType:
        """Run simulations and return the best action according to the policy.
        This is a concrete implementation that uses the abstract methods above.
        """

        # If the root node is a leaf node, it is initialized as if it has been visited before during tree search
        if self.root.is_leaf():
            self.root.expand(state_dict=self.state_dict)
            evaluation = self.evaluate(self.root)
            self.update(self.root, None, evaluation)

        for _ in range(self.num_simulations):
            node = self.root
            path = []  # List of (node, action) pairs for backpropagation

            # Selection
            while not node.is_leaf():
                action = self.select(node)
                path.append((node, action))
                node = node.children[action]

            path.append(
                (node, None)
            )  # Leaf node action is None. TODO: Remove once we change to self.backpropagate

            # Expansion
            if not node.state.is_terminal:
                node.expand(state_dict=self.state_dict)

            # Evaluation
            evaluation = self.evaluate(node)

            # Backpropagation, including the leaf node
            # TODO: change to self.backpropagate(path, leaf_node, evaluation)
            for node, action in reversed(path):
                evaluation = self.update(node, action, evaluation)

        return self.policy()

    def update_root(self, actions: List[ActionType]) -> None:
        """Update the root node after committing to a sequence of actions."""
        for action in actions:
            self.root.expand(state_dict=self.state_dict, actions=[action])
            self.root = self.root.children[action]

    def set_root(self, state: State[ActionType, PlayerType]) -> None:
        """Set the root node to a new state."""
        if state in self.state_dict:
            self.root = self.state_dict[state]
        else:
            self.root = Node[ActionType, ValueType, PlayerType](state)
            self.state_dict[state] = self.root
