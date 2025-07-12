from typing import List, Dict, Optional, Tuple, Callable, Any, Type, Generic, Self, Literal, Mapping
import torch
from dataclasses import dataclass, field, asdict
from mcts_playground.state import State
from mcts_playground.agent import TreeAgent
from mcts_playground.model_interface import Model, ModelPredictor, ModelInitParams
from mcts_playground.data_structures import ReplayBuffer
from mcts_playground.types import ActionType, PlayerType,TargetType
from mcts_playground.algorithms import RandomAgent
from .evaluation import Evaluator
from .simulation import generate_trajectories
import time
import os
from wandb.sdk.wandb_run import Run
import wandb
import torch.multiprocessing as multiprocessing
from multiprocessing.synchronize import Lock
from multiprocessing.sharedctypes import Synchronized
import itertools
from .tensor_mapping import TensorMapping
from .training_adapter import TrainingAdapter, AlgorithmParams
import logging
from logging.handlers import RotatingFileHandler

@dataclass
class TrainerConfig(Generic[ActionType, ModelInitParams, PlayerType, TargetType]):
    # Core parameters
    model_architecture: Type[torch.nn.Module]
    model_params: ModelInitParams
    tensor_mapping: TensorMapping[ActionType, TargetType]
    training_adapter: TrainingAdapter[ActionType, TargetType]
    algorithm_params: AlgorithmParams
    create_initial_state: Callable[[], State[ActionType, Any]] # Can not be a lambda function or defined in a local scope because it needs to be picklable
    checkpoint_dir: str
    
    # Optimizer
    optimizer: Type[torch.optim.Optimizer]
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    lr_scheduler: Type[torch.optim.lr_scheduler.LRScheduler] | None = None
    lr_scheduler_params: Dict[str, Any] = field(default_factory=dict)
    lr_scheduler_frequency: int = 1  # Update scheduler every N training iterations
    
    # Replay Buffer
    buffer_max_size: int = 100_000
    
    # Actors
    num_actors: int = 1
    actor_opponents: List[Callable[[State[ActionType, Any]], TreeAgent]] = field(default_factory=list) # Empty list means self-play
    actor_refresh_rate_seconds: float = 10.0 # Min seconds between adding examples to queue and updating model weights
    actor_device: torch.device = torch.device('cpu')

    # Learner
    learning_batch_size: int = 32
    learning_min_new_examples_per_step: int = 1024
    learning_min_seconds_per_step: float = 0.0 # Min seconds per step through main loop
    learning_fraction_of_buffer_per_step: float = 1.0 # Fraction of buffer examples to use per step through main loop
    learning_min_buffer_size: int = 10_000
    learning_device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    
    # Evaluation
    evaluator: Evaluator[ActionType, PlayerType] | None = None
    evaluator_algorithm_params: AlgorithmParams | None = None
    evaluator_frequency_seconds: float = 0.0 # Min seconds between evaluations

    # Logging to wandb. Disabled with default values.
    wandb_run: Run | None = None # If not None, will log to this run.
    wandb_project: str | None = None # If wandb_run is None and this is not None, will create a new run in this project.
    wandb_run_name: str | None = None # If wandb_run is None and wandb_project is not None, will give the run this name.
    wandb_run_id: str | None = None # Will attempt to resume from this run id.
    wandb_use_watch: bool = True # Whether to watch model parameters and gradients
    wandb_watch_log_level: Literal["all", "gradients", "parameters"] = "all" # If watch_model is True, this is the level of detail to log
    wandb_watch_log_freq: int | None = None # If watch_model is True, this is the log frequency for model parameters. If None, will be set to int(learning_min_buffer_size * learning_fraction_of_buffer_examples_per_iteration / learning_batch_size)
    wandb_watch_log_graph: bool = True # If watch_model is True, this is whether to log the model graph
    wandb_save_artifacts: bool = True # If True, will log model, replay buffer, and checkpoints to wandb
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_to_file: bool = True
    log_file_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO" # DEBUG will log all moves and states in individual games
    log_format: str = "%(asctime)s | %(name)s | %(levelname)s: %(message)s"
    
    # Misc
    max_training_steps: int | None = None
    max_training_time_hours: float | None = None # If set, training will stop after this many hours.

    # Checkpointing
    checkpoint_frequency_hours: float | None = 0.5 # Min hours between checkpoints
    checkpoint_frequency_steps: int | None = None # Min steps between checkpoints

    # Resuming. Resuming from checkpoint will override load_model_from_path and load_replay_buffer_from_path.
    load_model_from_path: str | None = None # If not None, __init__ will load model from this path.
    load_replay_buffer_from_path: str | None = None # If not None, __init__ will load replay buffer from this path.
    resume_from_wandb_checkpoint: str | None = None # If not None, __init__ will attempt to resume from wandb run with this id.
    resume_from_checkpoint_path: str | None = None # If provided and not resuming from wandb_run, __init__ will attempt to load checkpoint from this path.
    resume_from_last_checkpoint: bool = False # If True, not resuming from wandb_run, and not resuming from checkpoint_path, __init__ will attempt to load checkpoint from checkpoint_dir/checkpoint.pt


def actor_worker(
        model_architecture: Type[torch.nn.Module],
        model_params: Mapping[str, Any],
        model_lock: Lock,
        shared_pytorch_model: torch.nn.Module,
        create_initial_state: Callable[[], State],
        tensor_mapping: TensorMapping,
        training_adapter: TrainingAdapter,
        algorithm_params: AlgorithmParams,
        examples_queue: multiprocessing.Queue,
        actor_refresh_rate_seconds: float,
        actor_opponents: List[Callable[[State], TreeAgent]],
        checkpoint_dir: str,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"],
        log_file_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"],
        log_format: str,
        log_to_file: bool,
        device: torch.device,
        actor_name: str,
    ):
    """Simulates games and adds training examples to the examples queue."""
    # Set the number of threads for PyTorch operations to 1.
    # See https://docs.pytorch.org/docs/stable/notes/multiprocessing.html#avoid-cpu-oversubscription
    torch.set_num_threads(1)

    # Model initialization
    model = Model(model_architecture, model_params, device)
    model_predictor = ModelPredictor(model, tensor_mapping)
    with model_lock:
        model.model.load_state_dict(shared_pytorch_model.state_dict())
    
    # Logging setup in subprocess
    logger = _initialize_logger(
        checkpoint_dir=checkpoint_dir,
        log_format=log_format,
        log_file=f'{actor_name}.log' if log_to_file else None,
        name=actor_name,
        log_level=log_level,
        log_file_level=log_file_level,
    )
    
    # Main loop
    logger.info(f"Starting main loop of actor process {actor_name} (Process ID: {os.getpid()})...")
    num_self_players = max(len(create_initial_state().players) - len(actor_opponents), 1)
    last_refresh_time = time.time()
    examples = []
    for game_count in itertools.count(1):
        try:
            start_time = time.time()
            trajectories = generate_trajectories(
                initial_state_creator=create_initial_state,
                trajectory_player_creators=[lambda state: training_adapter.create_tree_search(state.clone(), model_predictor, algorithm_params) for _ in range(num_self_players)],
                opponent_creators=actor_opponents,
                num_games=1,
                logger=logger
            )
            # Extract examples and add to list of examples using algorithm-specific extractor
            for trajectory in trajectories:
                examples += training_adapter.extract_examples(trajectory)
            logger.debug(f"Collected {len(examples)} examples from game {game_count} in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Error in game {game_count}: {e}", exc_info=True)
            time.sleep(60.0)
            continue

        # Add examples to queue and update model weights
        if time.time() - last_refresh_time > actor_refresh_rate_seconds:
            try:
                encoded_examples = tensor_mapping.encode_examples(examples, device)
                examples_queue.put(encoded_examples)
                logger.debug(f"Added {encoded_examples[0].shape[0]} examples to queue.")
            except Exception as e:
                logger.error(f"Failed to add examples to queue: {e}", exc_info=True)
            try:
                with model_lock:
                    model.model.load_state_dict(shared_pytorch_model.state_dict())
                logger.debug(f"Updated model weights")
            except Exception as e:
                logger.error(f"Failed to update model weights: {e}", exc_info=True)

            examples = []
            last_refresh_time = time.time()

def evaluator_worker(
        *,
        evaluator: Evaluator[ActionType, PlayerType],
        evaluator_stats_queue: multiprocessing.Queue,
        model_architecture: Type[torch.nn.Module],
        model_params: Mapping[str, Any],
        model_lock: Lock,
        shared_pytorch_model: torch.nn.Module,
        tensor_mapping: TensorMapping,
        training_adapter: TrainingAdapter,
        algorithm_params: AlgorithmParams,
        checkpoint_dir: str,
        device: torch.device,
        evaluator_frequency_seconds: float,
        log_format: str,
        log_to_file: bool,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"],
        log_file_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"],
        training_step: Synchronized,
    ):
    torch.set_num_threads(1)

    # Initialize model
    model = Model(model_architecture, model_params, device)
    model.model.eval()
    model_predictor = ModelPredictor(model, tensor_mapping)
    with model_lock:
        model.model.load_state_dict(shared_pytorch_model.state_dict())
    
    # Logging setup in subprocess
    logger = _initialize_logger(
        checkpoint_dir=checkpoint_dir,
        log_format=log_format,
        log_file=f'evaluator.log' if log_to_file else None,
        name=f"Evaluator",
        log_level=log_level,
        log_file_level=log_file_level,
    )

    # Main loop
    for evaluation_count in itertools.count(1):
        logger.info(f"Starting evaluation {evaluation_count}...")
        try:
            start_time = time.time()
            stats = {}
            with model_lock:
                model.model.load_state_dict(shared_pytorch_model.state_dict())
                stats['training_step'] = training_step.value
            eval_stats = evaluator(lambda state: training_adapter.create_tree_search(state, model_predictor, algorithm_params), logger=logger)
            stats.update(eval_stats)
            evaluator_stats_queue.put(stats)
            stats['time_to_evaluate'] = time.time() - start_time
            log_message = f"Completed evaluation {evaluation_count} of training step {stats['training_step']} model in {stats['time_to_evaluate']:.2f} seconds:"
            for opponent, results in eval_stats.items():
                log_message += f"\n    {opponent}: "
                log_message += ", ".join([f"{name}: {value:.4f}" for name, value in results.items()])
            logger.info(log_message)
        except Exception as e:
            logger.error(f"Error in evaluation {evaluation_count}: {e}", exc_info=True)
        finally:
            time_until_next_eval = evaluator_frequency_seconds - (time.time() - start_time)
            if time_until_next_eval > 0.0:
                logger.info(f"Sleeping for {time_until_next_eval:.2f} seconds until next evaluation...")
                time.sleep(time_until_next_eval)

class Trainer(Generic[ActionType, PlayerType, ModelInitParams, TargetType]):
    def __init__(
        self,
        config: TrainerConfig[ActionType, ModelInitParams, PlayerType, TargetType]
    ):
        self.config = config

        # Initialize checkpoint path
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(config.checkpoint_dir, 'checkpoint.pt')
        self.model_path = os.path.join(config.checkpoint_dir, 'model.pt')
        self.buffer_path = os.path.join(config.checkpoint_dir, 'buffer.pt')

        # Initialize logger
        self.logger = _initialize_logger(
            checkpoint_dir=config.checkpoint_dir,
            log_format=config.log_format,
            log_file=f'trainer.log' if config.log_to_file else None,
            name=f"Trainer",
            log_level=config.log_level,
            log_file_level=config.log_file_level
        )
        self.logger.info("Initializing trainer...")

        # Initialize learning model
        assert config.load_model_from_path is not None or config.model_params is not None, "Must specify either load_model_from_path or model_params"
        if config.load_model_from_path is not None:
            self.model = Model.from_file(
                model_architecture=config.model_architecture,
                path=config.load_model_from_path,
                device=config.learning_device
            )
            config.model_params = self.model.init_params
            self.logger.info(f"Loaded model from {config.load_model_from_path}.")
        else:
            self.model = Model(
                model_architecture=config.model_architecture,
                init_params=config.model_params,
                device=config.learning_device
            )

        # Set multiprocessing start method to spawn. This is more reliable, for example on VMs in the cloud.
        multiprocessing.set_start_method('spawn', force=True)
        self.logger.info(f"Set multiprocessing start method to spawn.")

        # Shared model state for actor-learner communication
        self.model_lock = multiprocessing.Lock()
        self.shared_pytorch_model = config.model_architecture(**config.model_params).to(config.actor_device).share_memory()
        self.shared_pytorch_model.load_state_dict(self.model.model.state_dict())
        self.training_step = multiprocessing.Value('i', 0)

        # Initialize optimizer
        self.optimizer = config.optimizer(self.model.model.parameters(), **config.optimizer_params)
        self.lr_scheduler = config.lr_scheduler(self.optimizer, **config.lr_scheduler_params) if config.lr_scheduler is not None else None

        # Initialize replay buffer
        if config.load_replay_buffer_from_path is not None:
            self.replay_buffer = ReplayBuffer.from_file(config.load_replay_buffer_from_path, device=config.learning_device)
        else:
            self.replay_buffer = ReplayBuffer(config.buffer_max_size, device=config.learning_device)

        # Initialize wandb run
        if config.wandb_run is not None:
            self.wandb_run = config.wandb_run
        elif config.wandb_project is not None:
            try:
                self.wandb_run = wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=asdict(config),
                    resume="must" if config.wandb_run_id is not None else None,
                    id=config.wandb_run_id
                )
                self.logger.info(f"Initialized wandb run {self.wandb_run.project}/{self.wandb_run.name} with id {self.wandb_run.id}.")
            except Exception as e:
                self.logger.error(f"Error initializing wandb run with settings config.wandb_project={config.wandb_project}, config.wandb_run_name={config.wandb_run_name}, config.wandb_run_id={config.wandb_run_id}: {e}", exc_info=True)
                self.wandb_run = None
        else:
            self.wandb_run = None
        
        # Configure wandb_run if it exists
        if self.wandb_run is not None:
            if config.wandb_use_watch:
                self.wandb_run.watch(
                    self.model.model, 
                    log=config.wandb_watch_log_level, 
                    log_freq=config.wandb_watch_log_freq or int(config.buffer_max_size * config.learning_fraction_of_buffer_per_step / config.learning_batch_size), # Once per step once replay buffer is full
                    log_graph=config.wandb_watch_log_graph
                )
            self.wandb_run.define_metric("training_step") # Needed to log evaluation stats to correct (not latest) training step
            self.wandb_run.define_metric("*", step_metric="training_step")

        # Try to resume from checkpoint if specified, prioritizing resuming from wandb run
        checkpoint = None
        if config.resume_from_wandb_checkpoint is not None:
            try:
                assert self.wandb_run is not None, "Wandb run failed to initialize, so cannot resume from wandb run."
                artifact = self.wandb_run.use_artifact(f'{self.wandb_run.name}_checkpoint:latest')
                checkpoint = torch.load(f"{artifact.download()}/checkpoint.pt", map_location=config.learning_device)
                self.logger.info(f"Loaded checkpoint from wandb run {self.wandb_run.name}.")
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint using artifact from wandb run: {e}", exc_info=True)
                raise e
        if checkpoint is None and config.resume_from_checkpoint_path is not None:
            try:
                checkpoint = torch.load(config.resume_from_checkpoint_path, map_location=config.learning_device)
                self.logger.info(f"Loaded checkpoint from {self.checkpoint_path}.")
            except Exception as e:
                self.logger.error(f"Error loading checkpoint from file: {e}", exc_info=True)
                raise e
        if checkpoint is None and config.resume_from_last_checkpoint:
            if os.path.exists(self.checkpoint_path):
                try:
                    checkpoint = torch.load(self.checkpoint_path, map_location=config.learning_device)
                    self.logger.info(f"Loaded checkpoint from {self.checkpoint_path}.")
                except Exception as e:
                    self.logger.error(f"Error loading checkpoint from file: {e}.", exc_info=True)
                    raise e
            else:
                self.logger.warning(f"No checkpoint found at {self.checkpoint_path}. Training will start from scratch.")
        if checkpoint is not None:
            try:
                self._load_checkpoint(checkpoint)
            except Exception as e:
                self.logger.error(f"Error loading checkpoint: {e}", exc_info=True)
                raise e
            else:
                self.logger.info(f"Resumed from checkpoint.")

        self.logger.info(f"Trainer initialized")

    # TODO: Validate config?

    def _train_batch(self):
        """Train model on a batch. Returns a dictionary of metrics."""
        # TODO: Instead of sampling batches, divide buffer into batches and train on each batch.
        states, targets, extra_data = self.replay_buffer.sample(self.config.learning_batch_size)
        self.model.model.train()
        # autocast is a context manager that enables automatic mixed precision.
        # On CPU, it uses bfloat16 for AMX. On CUDA, it uses float16 for Tensor Cores.
        with torch.autocast(device_type=self.config.learning_device.type):
            model_outputs = self.model.model(states)
            loss, metrics = self.config.training_adapter.compute_loss(model_outputs, targets, extra_data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item(), **(metrics or {})}

    def __call__(self):
        """
        Orchestrates the main training loop using a parallel actor-learner setup.
        """
        config = self.config
        processes = []
        last_examples_time = start_time = time.time()

        try:
            # Initialize actor processes
            self.logger.info("Initializing actor processes...")
            examples_queue = multiprocessing.Queue()
            actor_processes = self._run_actor_subprocesses(config.num_actors, examples_queue)
            processes.extend(actor_processes)

            # Initialize evaluator process
            if self.config.evaluator is not None:
                self.logger.info("Initializing evaluator process...")
                evaluator_stats_queue = multiprocessing.Queue()
                evaluator_process = self._run_evaluator_subprocess(evaluator_stats_queue)
                processes.append(evaluator_process)
                time.sleep(3.0) # Some time for evaluator to start initial evaluation at step 0

            # Start main training loop
            self.logger.info("Starting main training loop..." if self.training_step.value == 0 else f"Resuming training from step {self.training_step.value + 1}...")
            last_checkpoint_time = last_step_time = time.time()
            log_dict = {}
            total_examples_in_session = 0
            for step in itertools.count(self.training_step.value + 1):
                self.logger.info(f"Training step {step}. Time elapsed: {(time.time() - start_time)/3600:.2f} hours. Current buffer size: {len(self.replay_buffer)}.")

                # Collect new examples from actors
                new_examples_count = 0
                while True:
                    while not examples_queue.empty():
                        encoded_examples = examples_queue.get_nowait()
                        self.replay_buffer.add(*encoded_examples)
                        new_examples_count += encoded_examples[0].shape[0]
                    if new_examples_count < config.learning_min_new_examples_per_step \
                        or len(self.replay_buffer) < config.learning_min_buffer_size:
                        self.logger.debug(f"Collected {new_examples_count} examples from actors. New examples this step: {new_examples_count}. Current buffer size: {len(self.replay_buffer)}. Waiting to reach {config.learning_min_new_examples_per_step} new examples and {config.learning_min_buffer_size} buffer size. Sleeping for 1 second...")
                        time.sleep(1.0)
                    else:
                        break
                log_dict['examples_per_second'] = new_examples_count / (time.time() - last_examples_time)
                last_examples_time = time.time()
                log_dict['buffer_size'] = len(self.replay_buffer)
                total_examples_in_session += new_examples_count
                log_dict['total_examples_in_session'] = total_examples_in_session
                self.logger.info(f"Collected {new_examples_count} new examples from actors. Rate of new examples: {log_dict['examples_per_second']:.2f} examples/s.")

                # Train on replay buffer
                temp_start_time = time.time()
                metrics_sum = {}
                for batch_count in itertools.count(1):
                    batch_metrics = self._train_batch()
                    for name, value in batch_metrics.items():
                        metrics_sum[name] = metrics_sum.get(name, 0.0) + value
                    if batch_count * config.learning_batch_size >= config.learning_fraction_of_buffer_per_step * len(self.replay_buffer):
                        log_dict['batches_trained_on'] = batch_count
                        break
                log_dict['learning_time'] = time.time() - temp_start_time
                metrics = {name: value / batch_count for name, value in metrics_sum.items()}
                log_dict.update(metrics)
                self.logger.info(f"Trained on {log_dict['batches_trained_on']} batches, covering {log_dict['batches_trained_on'] * config.learning_batch_size / len(self.replay_buffer):.2%} of the buffer, in {log_dict['learning_time']:.2f} seconds. Statistics:\n    {', '.join(f'{name}: {value:.4f}' for name, value in metrics.items())}")
                
                # Update learning rate
                if self.lr_scheduler and step > 0 and step % config.lr_scheduler_frequency == 0:
                    last_lr = self.lr_scheduler.get_last_lr()[0]
                    if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step(metrics['loss'])
                    else:
                        self.lr_scheduler.step()
                    new_lr = self.lr_scheduler.get_last_lr()[0]
                    log_dict['lr'] = new_lr
                    if new_lr != last_lr:
                        self.logger.info(f"Updated learning rate with {self.lr_scheduler.__class__.__name__}. New learning rate: {new_lr:.6f}")

                # Update shared model weights
                with self.model_lock:
                    self.shared_pytorch_model.load_state_dict(self.model.model.state_dict())
                    self.training_step.value += 1
                self.logger.debug(f"Shared new model weights")

                # Log metrics to wandb
                if self.wandb_run is not None:
                    try:
                        log_dict['training_step'] = step
                        self.wandb_run.log(log_dict)
                        self.logger.info(f"Logged metrics to wandb")
                    except Exception as e:
                        self.logger.error(f"Error logging metrics to wandb: {e}", exc_info=True)

                # Log evaluator stats to wandb
                if self.config.evaluator is not None \
                    and not evaluator_stats_queue.empty() \
                    and self.wandb_run is not None:
                    try:
                        eval_stats = evaluator_stats_queue.get_nowait()
                        self.wandb_run.log(eval_stats)
                        self.logger.info(f"Logged evaluator stats to wandb")
                    except Exception as e:
                        self.logger.error(f"Error collecting stats from evaluator: {e}", exc_info=True)

                # Save checkpoint
                if (config.checkpoint_frequency_hours is not None and time.time() - last_checkpoint_time > config.checkpoint_frequency_hours * 3600) \
                    or (config.checkpoint_frequency_steps is not None and step % config.checkpoint_frequency_steps == 0):
                    try:
                        self._save_checkpoint()
                        last_checkpoint_time = time.time()
                        self.logger.info(f"Saved checkpoint to {self.checkpoint_path}")
                    except Exception as e:
                        self.logger.error(f"Error saving checkpoint: {e}", exc_info=True)

                # Check exit conditions
                if config.max_training_steps is not None and step >= config.max_training_steps:
                    self.logger.info(f"Reached maximum number of training steps ({config.max_training_steps}).")
                    break
                if config.max_training_time_hours is not None and time.time() - start_time > config.max_training_time_hours * 3600:
                    self.logger.info(f"Reached maximum training time ({config.max_training_time_hours} hours).")
                    break

                # Check if we have reached the minimum seconds per step
                if time.time() - last_step_time < config.learning_min_seconds_per_step:
                    sleep_time = max(0.0, config.learning_min_seconds_per_step - (time.time() - last_step_time))
                    self.logger.debug(f"Have not reached minimum seconds per step ({config.learning_min_seconds_per_step} seconds). Sleeping for {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                last_step_time = time.time()

        except KeyboardInterrupt:
            self.logger.warning(f"Training interrupted by user.")
        finally:
            self.logger.info(f"Shutting down processes...")
            for p in processes:
                if p.is_alive():
                    p.terminate()
                p.join(timeout=10.0)

            self.logger.info(f"Saving final checkpoint, model, and replay buffer to {self.checkpoint_path}, {self.model_path}, and {self.buffer_path}, respectively...")
            self._save_checkpoint()
            self.model.save_to_file(self.model_path)
            self.replay_buffer.save_to_file(self.buffer_path)
            
            if self.wandb_run is not None and self.config.wandb_save_artifacts:
                self.model.save_to_wandb(wandb_run=self.wandb_run, model_name=f"{self.wandb_run.name}_model")
                self.replay_buffer.save_to_wandb(wandb_run=self.wandb_run, artifact_name=f"{self.wandb_run.name}_buffer")
                wandb.finish()

            self.logger.info(f"Shutdown complete.")

    def _save_checkpoint(self):
        if self.replay_buffer.current_size == 0:
            self.logger.warning("Replay buffer is empty. Skipping checkpoint save.")
            return
        
        checkpoint_data = {
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else {},
            'replay_buffer': {
                'states': self.replay_buffer.states[:self.replay_buffer.current_size], # type: ignore
                'targets': {key: self.replay_buffer.targets[key][:self.replay_buffer.current_size] for key in self.replay_buffer.targets.keys()}, # type: ignore
                'extra_data': {key: self.replay_buffer.extra_data[key][:self.replay_buffer.current_size] for key in self.replay_buffer.extra_data.keys()}, # type: ignore
                'max_size': self.replay_buffer.max_size,
                'write_index': self.replay_buffer.write_index
            },
            'training_step': self.training_step.value
        }
        torch.save(checkpoint_data, self.checkpoint_path)

        if self.wandb_run is not None and self.config.wandb_save_artifacts:
            artifact = wandb.Artifact(
                name=f'{self.wandb_run.name}_checkpoint',
                type='checkpoint',
                description='Checkpoint of the training run'
            )
            artifact.add_file(self.checkpoint_path)
            self.wandb_run.log_artifact(artifact)

    def _load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Load data from checkpoint into trainer."""
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        with self.model_lock:
            self.shared_pytorch_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.lr_scheduler is not None and checkpoint.get('lr_scheduler_state_dict'):
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.replay_buffer = ReplayBuffer(**checkpoint['replay_buffer'], device=self.config.learning_device)
        self.training_step.value = checkpoint['training_step']

    def _run_actor_subprocesses(self, num_actors: int, examples_queue: multiprocessing.Queue):
        """Run actor processes to generate training examples."""
        processes = []
        config = self.config
        for i in range(num_actors):
            actor_process = multiprocessing.Process(
                target=actor_worker,
                kwargs={
                    "model_architecture": config.model_architecture,
                    "model_params": config.model_params,
                    "model_lock": self.model_lock,
                    "shared_pytorch_model": self.shared_pytorch_model,
                    "create_initial_state": config.create_initial_state,
                    "tensor_mapping": config.tensor_mapping,
                    "training_adapter": config.training_adapter,
                    "algorithm_params": config.algorithm_params,
                    "examples_queue": examples_queue,
                    "actor_refresh_rate_seconds": config.actor_refresh_rate_seconds,
                    "actor_opponents": config.actor_opponents,
                    "checkpoint_dir": config.checkpoint_dir,
                    "log_level": config.log_level,
                    "log_format": config.log_format,
                    "log_to_file": config.log_to_file,
                    "log_file_level": config.log_file_level,
                    "device": config.actor_device,
                    "actor_name": f"actor-{i}"
                },
                name=f"Actor-{i}"
            )
            actor_process.start()
            processes.append(actor_process)
        return processes
    
    def _run_evaluator_subprocess(self, evaluator_stats_queue: multiprocessing.Queue):
        """Run evaluator process to evaluate the model."""
        config = self.config
        evaluator_process = multiprocessing.Process(
            target=evaluator_worker,
            kwargs={
                "evaluator": config.evaluator,
                "evaluator_stats_queue": evaluator_stats_queue,
                "model_architecture": config.model_architecture,
                "model_params": config.model_params,
                "model_lock": self.model_lock,
                "shared_pytorch_model": self.shared_pytorch_model,
                "tensor_mapping": config.tensor_mapping,
                "training_adapter": config.training_adapter,
                "algorithm_params": config.evaluator_algorithm_params,
                "checkpoint_dir": config.checkpoint_dir,
                "device": config.actor_device,
                "evaluator_frequency_seconds": config.evaluator_frequency_seconds,
                "log_format": config.log_format,
                "log_to_file": config.log_to_file,
                "log_level": config.log_level,
                "log_file_level": config.log_file_level,
                "training_step": self.training_step,
            },
            name="Evaluator"
        )
        evaluator_process.start()
        return evaluator_process
    
    def run_actors(self, num_actors: int, target_buffer_size: int):
        """Run actor processes to generate training examples.
        This function runs the actors and adds examples to the replay buffer _without_ training the model on the replay buffer.
        It is not called by __call__."""
        try:
            examples_queue = multiprocessing.Queue()
            processes = self._run_actor_subprocesses(num_actors, examples_queue)
            new_examples_count = 0
            while len(self.replay_buffer) < target_buffer_size:
                sleep_time = 60.0
                time.sleep(sleep_time)
                while not examples_queue.empty():
                    encoded_examples = examples_queue.get_nowait()
                    self.replay_buffer.add(*encoded_examples)
                    new_examples_count += encoded_examples[0].shape[0]
                self.logger.info(f"Collected {new_examples_count} examples from actors. Current buffer size: {len(self.replay_buffer)}.")
        except KeyboardInterrupt:
            self.logger.warning(f"Training interrupted by user.")
        finally:
            self.logger.info(f"Shutting down processes...")
            for p in processes:
                if p.is_alive():
                    p.terminate()
                p.join(timeout=10.0)
            self.replay_buffer.save_to_file(self.buffer_path)
            self.logger.info(f"Shutdown complete.")
    
    def run_evaluator(self):
        """Run evaluator process to evaluate the model. This is not called by __call__."""
        evaluator_stats_queue = multiprocessing.Queue() # Currently not used for anything in this method
        evaluator_process = self._run_evaluator_subprocess(evaluator_stats_queue)
        eval_stats = {}
        try:
            while True:
                if not evaluator_stats_queue.empty():
                    eval_stats = evaluator_stats_queue.get_nowait()
                    break
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.logger.warning(f"Evaluator interrupted by user.")
        finally:
            evaluator_process.terminate()
            evaluator_process.join(timeout=10.0)
            self.logger.info(f"Evaluator process terminated.")
            return eval_stats

def _initialize_logger(
        *,
        checkpoint_dir: str, log_format: str,
        log_file: str | None,
        name: str, 
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"],
        log_file_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    ) -> logging.Logger:
    """
    Initialize a logger with console and file handlers, using rotating file handler 
    to prevent log files from growing too large.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(getattr(logging, log_level.upper()))
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # Rotating file handler to prevent log files from growing too large
    if log_file is not None:
        file_handler = RotatingFileHandler(
            filename=os.path.join(checkpoint_dir, log_file),
            maxBytes=10 * 1024 * 1024,  # 10MB max file size
            backupCount=1  # Keep 1 backup file (log.1)
        )
        file_handler.setLevel(getattr(logging, log_file_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    return logger