from typing import Generic, Optional, Self, Type, Dict, Any
import torch
from core.tensor_mapping import TensorMapping
from core.tree_search import State
from core.types import ActionType, TargetType, ModelInitParams
import os
import wandb
from wandb.sdk.wandb_run import Run

class ModelInterface(Generic[ModelInitParams]):
    """Protocol for (deep learning) models used in tree search.
    
    This interface acts as a layer between tree search algorithms and PyTorch models.
    Its main responsibility is converting between game states and PyTorch tensors.

    The model attribute gives direct access to the underlying PyTorch model for forward(),
    parameters(), etc.
    """
    model: torch.nn.Module
    init_params: ModelInitParams

    def __init__(self, model_architecture: Type[torch.nn.Module], init_params: ModelInitParams, device: torch.device):
        self.model = model_architecture(**init_params)
        self.model.to(device)
        self.model.eval()
        self.init_params = init_params

    def predict(self, tensor_mapping: TensorMapping[ActionType, TargetType], state: State[ActionType]) -> TargetType:
        """Convenience function for single-state inference."""
        # We add the batch dimension before model inference and remove it after.
        device = next(self.model.parameters()).device
        encoded_state = tensor_mapping.encode_state(state, device).unsqueeze(0)
        outputs = self.model(encoded_state)
        outputs = {k: v.squeeze(0) for k, v in outputs.items()}
        return tensor_mapping.decode_output(outputs, state)
    
    def save_checkpoint(self, path: str, metadata: Dict[str, Any] = {}) -> None:
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
            self.save_checkpoint(path, metadata)
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
        run_id: Optional[str] = None,
        model_version: str = "latest",
        device: Optional[torch.device] = None,
    ) -> Self:
        """Initialize a model from a wandb artifact.
        
        Args:
            project: Wandb project name
            model_name: Name of the model
            artifact_dir: str,
            run_id: Optional run ID
            model_version: Version of the model to load
            device: Device to load the model onto
        """
        try:
            api = wandb.Api()
            if run_id:
                artifact = api.artifact(f'{project}/{run_id}/{model_name}:{model_version}')
            else:
                artifact = api.artifact(f'{project}/{model_name}:{model_version}')
            artifact_dir = artifact.download(root=artifact_dir)
            model_interface = cls.from_file(
                model_architecture=model_architecture,
                path=os.path.join(artifact_dir, f'{model_name}.pt'),
                device=device
            )
            return model_interface
        except Exception as e:
            print(f"Error loading model from wandb: {e}")
            raise e
