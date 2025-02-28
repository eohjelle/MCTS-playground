import os
import wandb
from wandb.sdk.wandb_run import Run
import torch
from typing import Any
from core.model_interface import ModelInterface
from core.data_structures import ReplayBuffer

def init_wandb(
    *,
    config: dict[str, Any],
    model: ModelInterface,
    wandb_project: str,
    wandb_dir: str,
    artifacts_dir: str,
    resume_id: str | None = None,
    watch: bool = False,
    log: str = 'all',
    log_freq: int = 100,
    log_graph: bool = True,
) -> tuple[Run, ReplayBuffer | None]:
    """Initialize a wandb run and optionally load a replay buffer."""
    
    # Create wandb directory if it doesn't exist
    wandb_dir = os.path.abspath(wandb_dir)
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ['WANDB_DIR'] = wandb_dir # Used by wandb to determine where to save files
    print(f"Using wandb directory: {wandb_dir}")
    
    # Get the starting iteration if resuming
    if resume_id:
        api = wandb.Api()
        try:
            previous_run = api.run(f"{wandb_project}/{resume_id}")
            # Get the last logged step
            history = previous_run.scan_history()
            steps = [row.get('_step', 0) for row in history]
            config['train_params']['start_iteration'] = max(steps) + 1 if steps else 0
            print(f"Resuming from iteration {config['train_params']['start_iteration']}")
        except Exception as e:
            print(f"Warning: Could not get previous iteration count: {e}")

    wandb_run = wandb.init(
        project=wandb_project,
        config=config,
        id=resume_id if resume_id else None,
        resume="must" if resume_id else "allow"
    )
    
    # Load artifacts if resuming a run
    if resume_id:
        assert artifacts_dir is not None, "artifacts_dir must be provided if use_wandb is True and wandb_resume_id is provided"

        # Create artifacts directory if it doesn't exist
        artifacts_dir = os.path.abspath(artifacts_dir)
        run_artifacts_dir = os.path.join(artifacts_dir, resume_id)
        os.makedirs(run_artifacts_dir, exist_ok=True)
        print(f"Using artifacts directory: {run_artifacts_dir}")
        
        # Load the final model artifact
        model_artifact = wandb_run.use_artifact(f'{config["model_name"]}:latest')
        model_path = model_artifact.download(root=run_artifacts_dir)
        model.load_checkpoint(os.path.join(model_path, f'{config["model_name"]}.pt'))
        print(f"Loaded model from wandb artifact")
        print(model.model)
        
        # Load the replay buffer artifact
        buffer_artifact = wandb_run.use_artifact(f'replay_buffer:latest')
        buffer_path = buffer_artifact.download(root=run_artifacts_dir)
        replay_buffer = torch.load(os.path.join(buffer_path, 'replay_buffer.pt'))
        print(f"Loaded replay buffer from wandb artifact")
    else:
        replay_buffer = None
    
    if watch:
        wandb_run.watch(
            model.model,
            log=log,
            log_freq=log_freq,
            log_graph=log_graph
        )

    return wandb_run, replay_buffer