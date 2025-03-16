from dataclasses import dataclass, field
from typing import Dict, Generic, Any, Callable, List, Tuple, Optional, Self, Type
from core.tree_search import State
from core.types import ActionType, TargetType
import torch
from wandb.sdk.wandb_run import Run
import wandb
import os

@dataclass
class TrainingExample(Generic[ActionType, TargetType]):
    """A single training example from self-play."""
    state: State[ActionType]
    target: TargetType
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReplayBuffer:
    """Buffer storing training examples for model training."""
    max_size: int  # Buffer max size
    states: torch.Tensor = field(default_factory=lambda: torch.tensor([]))  # [buffer_size, ...state_shape]
    targets: Dict[str, torch.Tensor] = field(default_factory=dict)  # Dict of [buffer_size, ...target_key_shape] tensors
    data: Dict[str, torch.Tensor] = field(default_factory=dict)  # Dict of [buffer_size, ...data_key_shape] tensors

    def extend(
        self, 
        examples: List[TrainingExample], 
        state_encoder: Callable[[State], torch.Tensor],
        example_encoder: Callable[[TrainingExample], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
    ):
        """Extend the replay buffer with new examples."""
        states = torch.stack([
            state_encoder(ex.state) for ex in examples
        ])
        encoded_targets, encoded_data = zip(*[example_encoder(ex) for ex in examples])
        targets = {
            key: torch.stack([
                encoded_targets[i][key] for i in range(len(encoded_targets))
            ])
            for key in encoded_targets[0].keys()
        }
        data = {
            key: torch.stack([
                encoded_data[i][key] for i in range(len(encoded_data))
            ])
            for key in encoded_data[0].keys()
        }

        # Initialize buffer if it's empty, otherwise extend it
        if self.states.numel() == 0:
            self.states = states
            self.targets = targets
            self.data = data
        else:
            self.states = torch.cat([self.states, states])
            for key in self.targets.keys():
                self.targets[key] = torch.cat([self.targets[key], targets[key]])
            for key in self.data.keys():
                self.data[key] = torch.cat([self.data[key], data[key]])
        
        # Trim to max size if needed
        if self.states.shape[0] > self.max_size:
            self.states = self.states[-self.max_size:]
            for key in self.targets.keys():
                self.targets[key] = self.targets[key][-self.max_size:]
            for key in self.data.keys():
                self.data[key] = self.data[key][-self.max_size:]

    def save(self, path: str) -> None:
        """Save replay buffer to disk.
        
        Args:
            path: Path to save the replay buffer to
        """
        torch.save({
            'max_size': self.max_size,
            'states': self.states,
            'targets': self.targets,
            'data': self.data
        }, path)
    
    @classmethod
    def from_file(cls, path: str, device: Optional[torch.device] = None) -> Self:
        """Load replay buffer from disk to the specified device or default device.
        
        Args:
            path: Path to the saved replay buffer
            device: Target device to load the buffer onto. If None, will use the default device (cuda, mps, or cpu based on availability)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Load buffer to specified device
        checkpoint = torch.load(path, map_location=device)
        return cls(**checkpoint)

    def save_to_wandb(
        self,
        *,
        artifact_name: str,
        wandb_run: Run | None = None,
        project: str | None = None,
        description: str | None = None
    ) -> None:
        """Save the replay buffer as a wandb artifact."""
        try:
            run = wandb_run or wandb.init(project=project, job_type='create_artifact')
            artifact = wandb.Artifact(
                name=artifact_name,
                type='dataset',
                description=description
            )
            path = os.path.join(run.dir, f'{artifact_name}.pt')
            self.save(path)
            artifact.add_file(path)
            run.log_artifact(artifact)
            if wandb_run is None:
                run.finish()
        except Exception as e:
            print(f"Error saving replay buffer artifact to wandb: {e}")
            raise e
    
    @classmethod
    def from_wandb(
        cls,
        *,
        project: str,
        artifact_name: str,
        artifact_dir: Optional[str] = None,
        artifact_version: str = "latest",
        device: Optional[torch.device] = None,
    ) -> Self:
        """Initialize a replay buffer from a wandb artifact.
        
        Args:
            project: Wandb project name
            artifact_name: Name of the artifact
            artifact_dir: str,
            artifact_version: Version of the artifact to load
            device: Device to load the artifact onto
        """
        try:
            api = wandb.Api()
            artifact = api.artifact(f'{project}/{artifact_name}:{artifact_version}')
            artifact_dir = artifact.download(root=artifact_dir)
            replay_buffer = cls.from_file(
                path=os.path.join(artifact_dir, f'{artifact_name}.pt'),
                device=device
            )
            return replay_buffer
        except Exception as e:
            print(f"Error loading replay buffer from wandb: {e}")
            raise e
