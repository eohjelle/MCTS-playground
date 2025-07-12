import random
from typing import Dict
from ..types import ActionType

def temperature_adjusted_policy(
    scores: Dict[ActionType, float], 
    temperature: float = 1.0
) -> Dict[ActionType, float]:
    """
    Convert scores/counts to temperature-adjusted probabilities.
    
    Args:
        scores: Dictionary mapping actions to scores/visit counts  
        temperature: Temperature parameter (0.0 = greedy, >1.0 = more random)
        
    Returns:
        Dictionary mapping actions to probabilities
    """
    if temperature == 0.0:
        # Greedy: uniform over max-scoring actions
        max_score = max(scores.values())
        max_actions = [action for action, score in scores.items() if score == max_score]
        prob = 1.0 / len(max_actions)
        return {action: prob if action in max_actions else 0.0 for action in scores}
    else:
        # Temperature scaling
        adjusted = {action: score**(1/temperature) for action, score in scores.items()}
        total = sum(adjusted.values())
        return {action: adj/total for action, adj in adjusted.items()} if total > 0 else scores

def sample_from_policy(policy: Dict[ActionType, float]) -> ActionType:
    """Sample action from probability distribution."""
    actions, probs = zip(*policy.items())
    return random.choices(actions, weights=probs, k=1)[0]

def greedy_action_from_scores(scores: Dict[ActionType, float]) -> ActionType:
    """Select action with highest score, breaking ties randomly."""
    max_score = max(scores.values())
    max_actions = [action for action, score in scores.items() if score == max_score]
    return random.choice(max_actions) 