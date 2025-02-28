from typing import Protocol, Dict, runtime_checkable, Optional
import torch
from core.tree_search import State
from core.types import ActionType, TargetType
import os
import wandb

@runtime_checkable
class ModelInterface(Protocol[ActionType, TargetType]):
    """Protocol for (deep learning) models used in tree search.
    
    This interface acts as a layer between tree search algorithms and PyTorch models.
    Its main responsibility is converting between game states and PyTorch tensors.

    Note that the encode_state, decode_output, and encode_target methods are for 
    single (not batched) states/targets. The batched model input is a stacked tensor of 
    encoded states, and the batched model output is a dictionary of stacked tensors.

    The model attribute gives direct access to the underlying PyTorch model for:
    - forward() - model inference (must return a dict of tensors)
    - train()/eval() - setting training mode
    - parameters() - optimization
    """
    model: torch.nn.Module

    def encode_state(self, state: State[ActionType]) -> torch.Tensor:
        """Convert a single state to model input tensor."""
        ...

    def decode_output(self, output: Dict[str, torch.Tensor], state: State) -> TargetType:
        """Convert raw model outputs to game-specific target format.
        
        This is used during inference to convert model outputs (dictionary of tensors)
        into a format the game understands (e.g. dictionary of action probabilities).
        """
        ...
    
    def encode_target(self, target: TargetType) -> Dict[str, torch.Tensor]:
        """Convert a target into tensor format for loss computation.
        
        This converts game-specific targets (e.g. dictionaries of action probabilities)
        into a dictionary of tensors that can be compared with model outputs.
        """
        ...
    
    def predict(self, state: State[ActionType]) -> TargetType:
        """Convenience method for single-state inference."""
        # We add the batch dimension before model inference and remove it after.
        encoded_state = self.encode_state(state).unsqueeze(0)
        outputs = self.model(encoded_state)
        outputs = {k: v.squeeze(0) for k, v in outputs.items()}
        return self.decode_output(outputs, state)
    
    def save_checkpoint(self, path: str) -> None:
        """Default implementation to save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'device': next(self.model.parameters()).device
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Default implementation to load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(checkpoint['device'])
        self.model.eval()

    def load_from_wandb_artifact(
        self,
        model_name: str,
        project: str,
        root_dir: str,
        run_id: Optional[str] = None,
        model_version: str = "latest",
    ) -> None:
        """Load a model from wandb artifacts."""
        api = wandb.Api()
        if run_id:
            artifact = api.artifact(f'{project}/{run_id}/{model_name}:{model_version}')
        else:
            artifact = api.artifact(f'{project}/{model_name}:{model_version}')
        artifact_dir = artifact.download(root=root_dir)
        self.load_checkpoint(os.path.join(artifact_dir, f'{model_name}.pt'))
