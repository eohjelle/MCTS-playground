import unittest
from AlphaZero.tree_search_algorithm import State, Predictor, Node, TreeSearchAlgorithm

# Dummy implementations for testing

class TerminalState(State):
    def __init__(self, value):
        # Terminal state always
        super().__init__(terminal=True, value=value)
    
    def childStates(self):
        # No children for terminal state
        return []

class NonTerminalState(State):
    def __init__(self, children_values=None):
        # Non-terminal state: children_values is a list of values for terminal children
        super().__init__(terminal=False)
        # if children_values are provided, we create dummy terminal states when childStates() is called
        self.children_values = children_values or []
    
    def childStates(self):
        # Return a list of TerminalState objects with given values
        return [TerminalState(val) for val in self.children_values]

class FakePredictor(Predictor):
    def predict(self, state: State):
        # For a non terminal state, return an arbitrary dummy policy and value.
        # The policy returned here is intended to be a dict mapping child Node -> probability.
        # In our tests we will not use the policy dict in UCT since tests call evaluate directly.
        # For simplicity, return an empty dict and an arbitrary value.
        return ({}, 3.14)

class TestTreeSearch(unittest.TestCase):
    def test_evaluate_terminal(self):
        # Create a terminal state with a known value and evaluate its node.
        term = TerminalState(value=42)
        node = Node(parent=None, children=[], value=0, policy=None, state=term)
        # Create a dummy predictor (won't be used because state is terminal)
        predictor = FakePredictor()
        tree = TreeSearchAlgorithm(root=node, predictor=predictor)
        tree.evaluate(node)
        # Expect that node.value is set to the terminal state's value.
        self.assertEqual(node.value, 42)
        
    def test_evaluate_non_terminal(self):
        # Create a non-terminal state.
        non_term = NonTerminalState()
        node = Node(parent=None, children=[], value=0, policy=None, state=non_term)
        predictor = FakePredictor()
        tree = TreeSearchAlgorithm(root=node, predictor=predictor)
        tree.evaluate(node)
        # Expect that evaluate uses the predictor value.
        self.assertEqual(node.value, 3.14)
        # And that the policy has been set (even if empty in this fake predictor)
        self.assertEqual(node.policy, {})

    def test_expand(self):
        # Create a non-terminal state that returns two terminal child states with values 1 and 2.
        non_term = NonTerminalState(children_values=[1, 2])
        node = Node(parent=None, children=[], value=0, policy=None, state=non_term)
        predictor = FakePredictor()
        tree = TreeSearchAlgorithm(root=node, predictor=predictor)
        tree.expand(node)
        # After expansion, number of children should equal number of child states.
        self.assertEqual(len(node.children), 2)
        # Check that children's states are terminal and have the expected values.
        child_values = [child.state.value for child in node.children]
        self.assertCountEqual(child_values, [1, 2])
    
    def test_backpropagate(self):
        # Create a simple two-level tree (parent and one child)
        parent_state = TerminalState(value=0)  # value here is dummy and will be overridden by backpropagation
        child_state = TerminalState(value=10)
        parent_node = Node(parent=None, children=[], value=0, policy=None, state=parent_state)
        child_node = Node(parent=parent_node, children=[], value=0, policy=None, state=child_state)
        # Manually add the child to the parent's children list.
        parent_node.children.append(child_node)
        
        predictor = FakePredictor()
        tree = TreeSearchAlgorithm(root=parent_node, predictor=predictor)
        
        # Perform backpropagation starting from the child.
        tree.backpropagate(child_node, 10)
        
        # The backpropagation loop updates the starting node and its ancestors.
        # For the child_node, visitCount should become 1 and value updated to 10.
        self.assertEqual(child_node.visitCount, 1)
        self.assertEqual(child_node.value, 10)
        # Then parent's visitCount will also be incremented by 1 and its value set to 10.
        self.assertEqual(parent_node.visitCount, 1)
        self.assertEqual(parent_node.value, 10)

if __name__ == '__main__':
    unittest.main() 