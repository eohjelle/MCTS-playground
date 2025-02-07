import unittest
from core.tree_search_algorithm import State, Selector, Evaluator, Node, TreeSearchAlgorithm, UCTSelector

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

class FakeEvaluator(Evaluator):
    def evaluate(self, state: State):
        # For a non-terminal state, return a dummy policy and value.
        # For simplicity, return an empty dict and an arbitrary value.
        return ({}, 3.14)

class TestTreeSearch(unittest.TestCase):
    def test_evaluate_terminal(self):
        # Create a terminal state with a known value and evaluate its node.
        term = TerminalState(value=42)
        node = Node(parent=None, children=[], value=0, policy=None, state=term)
        # Create a dummy evaluator (won't be used because state is terminal)
        evaluator = FakeEvaluator()
        tree = TreeSearchAlgorithm(root=node, evaluator=evaluator)
        tree.evaluate(node)
        # Expect that node.value is set to the terminal state's value.
        self.assertEqual(node.value, 42)
        
    def test_evaluate_non_terminal(self):
        # Create a non-terminal state.
        non_term = NonTerminalState()
        node = Node(parent=None, children=[], value=0, policy=None, state=non_term)
        evaluator = FakeEvaluator()
        tree = TreeSearchAlgorithm(root=node, evaluator=evaluator)
        tree.evaluate(node)
        # Expect that evaluate uses the evaluator value.
        self.assertEqual(node.value, 3.14)
        # And that the policy has been set (even if empty in this fake evaluator)
        self.assertEqual(node.policy, {})

    def test_expand(self):
        # Create a non-terminal state that returns two terminal child states with values 1 and 2.
        non_term = NonTerminalState(children_values=[1, 2])
        node = Node(parent=None, children=[], value=0, policy=None, state=non_term)
        evaluator = FakeEvaluator()
        tree = TreeSearchAlgorithm(root=node, evaluator=evaluator)
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
        
        evaluator = FakeEvaluator()
        tree = TreeSearchAlgorithm(root=parent_node, evaluator=evaluator)
        
        # Perform backpropagation starting from the child node.
        tree.backpropagate(child_node, 10)
        
        # The backpropagation loop updates the starting node and its ancestors.
        # For the child_node, visitCount should become 1 and value updated to 10.
        self.assertEqual(child_node.visitCount, 1)
        self.assertEqual(child_node.value, 10)
        # Then parent's visitCount will also be incremented by 1 and its value set to 10.
        self.assertEqual(parent_node.visitCount, 1)
        self.assertEqual(parent_node.value, 10)

    def test_UCTSelector(self):
        import math
        # Set up a parent node with a dummy non-terminal state
        parent_state = NonTerminalState()
        parent_node = Node(parent=None, children=[], value=0, policy={}, state=parent_state)

        # Create two child nodes with dummy terminal states
        child1_state = TerminalState(1)
        child2_state = TerminalState(2)
        child1 = Node(parent=parent_node, children=[], value=10, policy=None, state=child1_state)
        child2 = Node(parent=parent_node, children=[], value=20, policy=None, state=child2_state)

        # Set visit counts for children and parent
        child1.visitCount = 5
        child2.visitCount = 10
        parent_node.visitCount = 15

        # Set parent's policy mapping for each child
        parent_node.policy[child1] = 0.5
        parent_node.policy[child2] = 0.8

        # Attach the children to the parent node
        parent_node.children.extend([child1, child2])

        # Instantiate UCTSelector and select a child
        selector = UCTSelector()
        selected_child = selector.select(parent_node)

        # Manually compute the UCT scores for each child
        uct1 = child1.value + parent_node.policy[child1] * math.sqrt(math.log(parent_node.visitCount + 1) / (child1.visitCount + 1))
        uct2 = child2.value + parent_node.policy[child2] * math.sqrt(math.log(parent_node.visitCount + 1) / (child2.visitCount + 1))
        expected_child = child1 if uct1 > uct2 else child2

        # Assert that the selector returns the child with the maximum UCT value
        self.assertEqual(selected_child, expected_child)

    def test_selector_instantiation(self):
        # Test that instantiating the abstract Selector directly raises a TypeError.
        with self.assertRaises(TypeError):
            Selector()

    def test_dummy_selector(self):
        # Create a dummy concrete implementation of Selector that always selects the first child.
        class DummySelector(Selector):
            def select(self, node: Node) -> Node:
                return node.children[0] if node.children else None

        # Set up a parent node with two children.
        dummy_state = TerminalState(0)
        parent = Node(parent=None, children=[], value=0, policy={}, state=dummy_state)
        child1 = Node(parent=parent, children=[], value=1, policy=None, state=TerminalState(1))
        child2 = Node(parent=parent, children=[], value=2, policy=None, state=TerminalState(2))
        parent.children.extend([child1, child2])

        dummy_selector = DummySelector()
        selected_child = dummy_selector.select(parent)
        self.assertEqual(selected_child, child1)

if __name__ == '__main__':
    unittest.main() 