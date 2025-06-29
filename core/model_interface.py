from typing import Generic, Optional, Self, Type, Dict, Any
import torch
from .state import State
from .tensor_mapping import TensorMapping
from .types import ActionType, TargetType, ModelInitParams
import os
import wandb
from wandb.sdk.wandb_run import Run

class Model(Generic[ModelInitParams]):
    """Wrapper for a PyTorch model. 
    Provides convenience functions for saving and loading models.
    """
    model: torch.nn.Module
    init_params: ModelInitParams

    def __init__(self, model_architecture: Type[torch.nn.Module], init_params: ModelInitParams, device: torch.device):
        self.model = model_architecture(**init_params)
        self.model.to(device)
        self.model.eval()
        self.init_params = init_params
    
    def save_to_file(self, path: str, metadata: Dict[str, Any] = {}) -> None:
        """Save model checkpoint."""
        torch.save({
            **metadata,
            'state_dict': self.model.state_dict(),
            'init_params': self.init_params
        }, path)

    @classmethod
    def from_file(
        cls,
        *,
        model_architecture: Type[torch.nn.Module],
        path: str, 
        device: Optional[torch.device] = None
    ) -> Self:
        """Load a model interface from a checkpoint file.
        
        Args:
            path: Path to the checkpoint file
            device: Device to load the model on (system-specific, not part of model config)
            
        Returns:
            A new model interface instance with the loaded model
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        checkpoint = torch.load(path, map_location=device)
        model_interface = cls(
            model_architecture=model_architecture,
            init_params=checkpoint['init_params'],
            device=device
        )
        model_interface.model.load_state_dict(checkpoint['state_dict'])
        model_interface.model.to(device)
        model_interface.model.eval()
        return model_interface
    
    def save_to_wandb(
        self,
        *,
        model_name: str,
        wandb_run: Run | None = None,
        project: str | None = None,
        description: str | None = None,
        metadata: Dict[str, Any] = {}
    ) -> None:
        """Save the model as a wandb artifact.
        
        Args:
            model_name: Name of the artifact
            wandb_run: Wandb run to save the artifact to. If None, will create a new run.
            project: Name of the wandb project to save the artifact to, if wandb_run is None.
            description: Description of the artifact
        """
        try:
            run = wandb_run or wandb.init(project=project, job_type='create_artifact')
            model_artifact = wandb.Artifact(
                name=model_name,
                type='model',
                description=description,
                metadata={
                    **metadata,
                    'model_architecture': self.model.__class__.__name__,
                    'init_params': self.init_params
                }
            )
            path = os.path.join(run.dir, f'{model_name}.pt')
            self.save_to_file(path, metadata)
            model_artifact.add_file(path)
            run.log_artifact(model_artifact)
            if wandb_run is None:
                run.finish()
        except Exception as e:
            print(f"Error saving model artifact to wandb: {e}")
            raise e
    
    @classmethod
    def from_wandb(
        cls,
        *,
        model_architecture: Type[torch.nn.Module],
        project: str,
        model_name: str,
        artifact_dir: Optional[str] = None,
        model_version: str = "latest",
        device: Optional[torch.device] = None,
    ) -> Self:
        """Initialize a model from a wandb artifact.
        
        Args:
            project: Wandb project name
            model_name: Name of the model
            artifact_dir: str,
            model_version: Version of the model to load
            device: Device to load the model onto
        """
        try:
            api = wandb.Api()
            artifact = api.artifact(f'{project}/{model_name}:{model_version}')
            artifact_dir = artifact.download(root=artifact_dir)
            model = cls.from_file(
                model_architecture=model_architecture,
                path=os.path.join(artifact_dir, f'{model_name}.pt'),
                device=device
            )
            return model
        except Exception as e:
            print(f"Error loading model from wandb: {e}")
            raise e


class ModelPredictor(Generic[ActionType, TargetType]):
    """Class that uses a model to make predictions."""

    def __init__(self, model: Model, tensor_mapping: 'TensorMapping[ActionType, TargetType]'):
        self.model = model
        self.tensor_mapping = tensor_mapping
    
    def __call__(self, state: State[ActionType, Any]) -> TargetType:
        """Convenience function for single-state inference."""
        # We add the batch dimension before model inference and remove it after.
        device = next(self.model.model.parameters()).device
        encoded_state = self.tensor_mapping.encode_states([state], device)
        # autocast is a context manager that enables automatic mixed precision.
        # On CPU, it uses bfloat16 for AMX. On CUDA, it uses float16 for Tensor Cores.
        with torch.autocast(device_type=device.type):
            outputs = self.model.model(encoded_state)
        decoded_outputs = self.tensor_mapping.decode_outputs(outputs, [state])
        return decoded_outputs[0]