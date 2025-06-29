from dataclasses import dataclass, field
from typing import Dict, Generic, Any, Tuple, Optional, Self, List
from core.tree_search import State, Node
from core.types import ActionType, TargetType
import torch
from torch import Size
from wandb.sdk.wandb_run import Run
import wandb
import os
import torch.multiprocessing as multiprocessing
import numpy as np

@dataclass
class TrainingExample(Generic[ActionType, TargetType]):
    """A single training example from self-play."""
    state: State[ActionType, Any]
    target: TargetType
    extra_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrajectoryStep(Generic[ActionType]):
    node: Node[ActionType, Any, Any]
    action: ActionType
    reward: float = 0.0

Trajectory = List[TrajectoryStep[ActionType]]

class ReplayBuffer:
    """Buffer storing training examples for model training."""

    def __init__(
        self,
        max_size: int,
        states: torch.Tensor | None = None,
        targets: Dict[str, torch.Tensor] | None = None,
        extra_data: Dict[str, torch.Tensor] | None = None,
        device: Optional[torch.device] = None,
        write_index: int | None = None,
    ):
        self.write_index = 0 # If supplying a write index manually for resuming training, have to set it _after_ loading the data into the buffer
        self.current_size = 0
        self.max_size = max_size
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        self.states = None # If not set, condition below can yield no attribute 'states' error
        self.targets, self.extra_data = {}, {}
        if states is not None:
            self.add(states, targets or {}, extra_data or {})

        if write_index is not None:
            assert write_index <= self.current_size, "Can not supply write index that is greater than current size."
            assert write_index < self.max_size, "Write index must be less than max size."
            self.write_index = write_index

    def __len__(self) -> int:
        """Return the number of examples in the buffer."""
        return self.current_size

    # TODO: Check why this seems to no be working (legal moves remain 0 for dots and boxes simple tensor mapping)
    def add(
        self, 
        states: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        extra_data: Dict[str, torch.Tensor]
    ):
        """Add new examples to the replay buffer."""
        # Initialize tensors as zeros of the correct shape if they are not already initialized
        if self.states is None:
            assert states.shape[0] > 0, "Cannot initialize with empty tensors."
            self.states = torch.empty(self.max_size, *states.shape[1:], dtype=states.dtype, device=self.device)
            self.targets = {key: torch.empty(self.max_size, *tensor.shape[1:], dtype=tensor.dtype, device=self.device) for key, tensor in targets.items()}
            self.extra_data = {key: torch.empty(self.max_size, *tensor.shape[1:], dtype=tensor.dtype, device=self.device) for key, tensor in extra_data.items()}

        # Move data to device and verify the number of examples
        states = states.to(self.device)
        length = states.shape[0]
        for key in targets.keys():
            assert targets[key].shape[0] == length, "Target tensors must have the same length as states."
            targets[key] = targets[key].to(self.device)
        for key in extra_data.keys():
            assert extra_data[key].shape[0] == length, "Extra data tensors must have the same length as states."
            extra_data[key] = extra_data[key].to(self.device)

        self.current_size = min(self.current_size + states.shape[0], self.max_size)

        start_idx = self.write_index
        left_over = states.shape[0]
        
        while left_over > 0:
            end_idx = min(start_idx + left_over, self.max_size)

            # Copy data to buffer
            self.states[start_idx:end_idx] = states[:end_idx - start_idx]
            for key in self.targets.keys():
                self.targets[key][start_idx:end_idx] = targets[key][:end_idx - start_idx]
            for key in self.extra_data.keys():
                self.extra_data[key][start_idx:end_idx] = extra_data[key][:end_idx - start_idx]

            # Update indices and data
            left_over = states.shape[0] - end_idx + start_idx
            if left_over == 0: # No more data to add, most common case
                break 
            else: # Cycle around
                start_idx = 0
                states = states[-left_over:]
                for key in targets.keys():
                    targets[key] = targets[key][-left_over:]
                for key in extra_data.keys():
                    extra_data[key] = extra_data[key][-left_over:]

        self.write_index = end_idx % self.max_size

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Sample a batch of data from the replay buffer."""
        if self.states is None or self.targets is None or self.extra_data is None:
            raise RuntimeError("Cannot sample from uninitialized replay buffer. Add data first.")
        
        indices = np.random.randint(0, self.current_size, batch_size)
        return self.states[indices], {key: self.targets[key][indices] for key in self.targets.keys()}, {key: self.extra_data[key][indices] for key in self.extra_data.keys()}

    def save_to_file(self, path: str) -> None:
        """Save replay buffer to disk.
        
        Args:
            path: Path to save the replay buffer to
        """
        end_idx = self.current_size
        torch.save({
            'max_size': self.max_size,
            'write_index': self.write_index,
            'states': self.states[:end_idx] if end_idx > 0 else None, # type: ignore
            'targets': {key: self.targets[key][:end_idx] for key in self.targets.keys()} if end_idx > 0 else None,
            'extra_data': {key: self.extra_data[key][:end_idx] for key in self.extra_data.keys()} if end_idx > 0 else None,
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
        instance = cls(**checkpoint, device=device)
        instance.write_index = checkpoint['write_index']
        return instance

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
            self.save_to_file(path)
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
        
# class SharedReplayBuffer(ReplayBuffer):
#     """Shared replay buffer for multiprocessing."""

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.lock = multiprocessing.Lock() 
#         self.shared_write_index = multiprocessing.Value('i', self.write_index)
#         self.shared_current_size = multiprocessing.Value('i', self.current_size)
#         assert self.device == torch.device('cpu'), "Shared replay buffer must be on CPU."
#         self.states = self.states.share_memory_()
#         for key in self.targets.keys():
#             self.targets[key] = self.targets[key].share_memory_()
#         for key in self.extra_data.keys():
#             self.extra_data[key] = self.extra_data[key].share_memory_()

#     def __len__(self) -> int:
#         with self.lock:
#             return self.shared_current_size.value

#     def add(self, *args, **kwargs):
#         with self.lock:
#             self.write_index = self.shared_write_index.value
#             self.current_size = self.shared_current_size.value
#             super().add(*args, **kwargs)
#             self.shared_write_index.value = self.write_index
#             self.shared_current_size.value = self.current_size

#     def sample(self, batch_size: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
#         with self.lock:
#             self.current_size = self.shared_current_size.value
#             return super().sample(batch_size)
        
#     def save_to_file(self, path: str) -> None:
#         with self.lock:
#             self.current_size = self.shared_current_size.value
#             super().save_to_file(path)