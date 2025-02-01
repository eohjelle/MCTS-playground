import unittest
from typing import List, Dict, Tuple
from tree_search import State, Predictor, Node, TreeSearch

# Simple implementations for testing
class TestState(State):
    def __init__(self, value: float = None, is_terminal: bool = False, children: List['TestState'] = None):
        super().__init__(value)
        self._is_terminal = is_terminal
        self._children = children or []

    def isTerminal(self) -> bool:
        return self._is_terminal

    def childStates(self) -> List['State']:
        return self._children

class TestPredictor(Predictor):
    def predict(self, node: State) -> Tuple[float, Dict[Node, float]]:
        # Always return 0.5 for value and uniform policy for simplicity
        return {}, 0.5

class TestTreeSearch(unittest.TestCase):
    def setUp(self):
        # Create a simple game tree for testing
        self.leaf1 = TestState(value=1.0, is_terminal=True)
        self.leaf2 = TestState(value=-1.0, is_terminal=True)
        self.root_state = TestState(children=[self.leaf1, self.leaf2])
        
        self.root_node = Node(
            parent=None,
            children=[],
            value=0,
            policy=None,
            state=self.root_state
        )
        
        self.predictor = TestPredictor()
        self.tree_search = TreeSearch(self.root_node, self.predictor)

    def test_node_initialization(self):
        node = Node(None, [], 0, None, self.root_state)
        self.assertIsNone(node.parent)
        self.assertEqual(node.children, [])
        self.assertEqual(node.value, 0)
        self.assertEqual(node.visitCount, 0)

    def test_node_is_leaf(self):
        node = Node(None, [], 0, None, self.root_state)
        self.assertTrue(node.isLeaf())
        
        node.children = [Node(node, [], 0, None, self.leaf1)]
        self.assertFalse(node.isLeaf())

    def test_expand(self):
        self.tree_search.expand(self.root_node)
        self.assertEqual(len(self.root_node.children), 2)
        self.assertTrue(all(isinstance(child, Node) for child in self.root_node.children))

    def test_evaluate_terminal_node(self):
        leaf_node = Node(None, [], 0, None, self.leaf1)
        self.tree_search.evaluate(leaf_node)
        self.assertEqual(leaf_node.value, 1.0)

    def test_evaluate_non_terminal_node(self):
        node = Node(None, [], 0, None, self.root_state)
        self.tree_search.evaluate(node)
        self.assertEqual(node.value, 0.5)  # From our TestPredictor

    def test_backpropagate(self):
        child = Node(self.root_node, [], 0, None, self.leaf1)
        self.root_node.children = [child]
        
        self.tree_search.backpropagate(child, 1.0)
        self.assertEqual(child.visitCount, 1)
        self.assertEqual(child.value, 1.0)

if __name__ == '__main__':
    unittest.main() 