import wandb
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from core.agent import Agent
from core.benchmark import benchmark
from core.model_interface import ModelInterface
from core.data_structures import ReplayBuffer, State
from core.tensor_mapping import TensorMapping
from core.implementations.AlphaZero import AlphaZeroModelAgent
from wandb.sdk.wandb_run import Run
from typing import Dict, Callable
import os

def baseline_best_loss(
    project: str,
    model_name: str
):
    api = wandb.Api()
    artifact = api.artifact(f'{project}/{model_name}:latest')
    return artifact.metadata['val_loss']
    

def supervised_training_loop(
    *,
    model_interface: ModelInterface,
    tensor_mapping: TensorMapping,
    buffer: ReplayBuffer,
    initial_state: Callable[[], State],
    evaluate_against_agents: Dict[str, Callable[[State], Agent]],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    wandb_run: Run | None,
    model_name: str,
    epochs: int,
    batch_size: int,
    mask_illegal_moves: bool = False,
    eval_freq: int,
    mask_value: float = -20.0,
    start_at: int = 1,
    checkpoint_dir: str,
    checkpoint_freq: int,
):
    """
    Train a model using a fixed dataset and a simple training loop. No tree search and no self-play involved.
    """
    model = model_interface.model

    # Create datasets
    states = buffer.states
    policy_targets = buffer.targets['policy']
    value_targets = buffer.targets['value']
    legal_actions_mask = buffer.data['legal_actions']
    
    # Create dataset and split into train/val
    dataset = TensorDataset(states, policy_targets, value_targets, legal_actions_mask)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize best validation loss
    if wandb_run is not None:
        try:
            best_val_loss = float(baseline_best_loss(wandb_run.project, f"{model_name}_best"))
        except Exception as e:
            print(f"Error getting baseline best loss: {e}")
            best_val_loss = float('inf')
    else:
        best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(start_at, start_at + epochs):
        # Training phase
        model.train()
        train_losses = []
        policy_losses = []
        value_losses = []
        
        for batch in train_loader:
            states_batch, policy_targets_batch, value_targets_batch, legal_actions_batch = batch
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(states_batch)
            policy_logits = outputs['policy']
            value_pred = outputs['value']
            
            # Apply mask for illegal moves if enabled
            if mask_illegal_moves:
                policy_logits = policy_logits * legal_actions_batch + (1 - legal_actions_batch) * mask_value
            
            # Compute losses
            policy_loss = F.cross_entropy(policy_logits, policy_targets_batch)
            value_loss = F.mse_loss(value_pred, value_targets_batch)
            total_loss = policy_loss + value_loss
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            
            # Track metrics
            train_losses.append(total_loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        val_policy_losses = []
        val_value_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                states_batch, policy_targets_batch, value_targets_batch, legal_actions_batch = batch
                
                # Forward pass
                outputs = model(states_batch)
                policy_logits = outputs['policy']
                value_pred = outputs['value']
                
                # Apply mask for illegal moves if enabled
                if mask_illegal_moves:
                    policy_logits = policy_logits * legal_actions_batch + (1 - legal_actions_batch) * mask_value
                
                # Compute losses
                policy_loss = F.cross_entropy(policy_logits, policy_targets_batch)
                value_loss = F.mse_loss(value_pred, value_targets_batch)
                total_loss = policy_loss + value_loss
                
                # Track metrics
                val_losses.append(total_loss.item())
                val_policy_losses.append(policy_loss.item())
                val_value_losses.append(value_loss.item())
        
        # Calculate average metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics to wandb
        if wandb_run is not None:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_policy_loss": np.mean(policy_losses),
                "train_value_loss": np.mean(value_losses),
                "val_loss": avg_val_loss,
                "val_policy_loss": np.mean(val_policy_losses),
                "val_value_loss": np.mean(val_value_losses),
                "learning_rate": current_lr
            }, step=epoch)
        
        print(f"Epoch {epoch}/{start_at + epochs - 1}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Learning rate: {current_lr:.6f}")
        
        # Update the learning rate scheduler
        if lr_scheduler is not None:

            # Update scheduler based on its actual type
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(avg_train_loss)  # ReduceLROnPlateau needs a metric
            else:
                lr_scheduler.step()  # All other schedulers just need step()
    
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Save model checkpoint
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
            metadata = {
                'val_loss': best_val_loss,
                'epoch': epoch
            }
            model_interface.save_checkpoint(
                checkpoint_path,
                metadata=metadata
            )
            if wandb_run is not None:
                model_interface.save_to_wandb(
                    model_name=f"{model_name}_best",
                    wandb_run=wandb_run,
                    description=f'Best model with val_loss: {best_val_loss:.4f}',
                    metadata=metadata
                )

        # Evaluate against agents
        if epoch % eval_freq == 0 or epoch == epochs:
            stats = benchmark(
                create_agent=lambda state: AlphaZeroModelAgent(
                    initial_state=state,
                    model=model_interface,
                    tensor_mapping=tensor_mapping,
                    temperature=0.0
                ),
                create_opponents=evaluate_against_agents,
                initial_state=initial_state,
                num_games=100
            )
            for agent_name, agent_stats in stats.items():
                print(f"{agent_name} score: {agent_stats['win_rate'] - agent_stats['loss_rate']}")
                if wandb_run is not None:
                    wandb.log({
                        f'{agent_name}_win_rate': agent_stats['win_rate'],
                        f'{agent_name}_draw_rate': agent_stats['draw_rate'],
                        f'{agent_name}_loss_rate': agent_stats['loss_rate'],
                        f'{agent_name}_score': agent_stats['win_rate'] - agent_stats['loss_rate']
                    }, step=epoch)

        # Save model checkpoint
        if epoch % checkpoint_freq == 0:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.pt")
            metadata = {
                'val_loss': avg_val_loss,
                'epoch': epoch
            }
            model_interface.save_checkpoint(
                checkpoint_path,
                metadata=metadata
            )
            if wandb_run is not None:
                model_interface.save_to_wandb(
                    model_name=model_name,
                    wandb_run=wandb_run,
                    description=f'Model at epoch {epoch} with val_loss: {avg_val_loss:.4f}',
                    metadata=metadata
                )

    print(f"Training complete.\nBest val_loss: {best_val_loss:.4f}\nFinal val_loss: {avg_val_loss:.4f}")