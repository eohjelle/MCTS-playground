**Disclaimer**: This is written by AI, summarizing a conversation. But I think it could be geniunely helpful for someone (like me) setting up cloud computing for the first time.

# Guide: From Laptop to the Cloud for Parallel Training

This document is a comprehensive tutorial for migrating a parallel Python-based machine learning workload from a local machine to a powerful, CPU-optimized Google Cloud Virtual Machine (VM). It captures not just the "how" (the commands) but also the "why" (the strategic decisions, hardware deliberations, and performance tuning insights).

---

## Part 1: The "Why" — Choosing Your Cloud Strategy

Before running any commands, it's crucial to decide on the right type of service.

### IaaS vs. PaaS: Google Compute Engine vs. Vertex AI

- **Infrastructure as a Service (IaaS)**, like **Google Compute Engine (GCE)**, gives you a raw virtual machine. You have full control over the operating system, dependencies, and environment. It's like having a powerful computer in the cloud.
  - **Pros:** Maximum flexibility, mirrors a local setup, easy to migrate existing scripts.
  - **Cons:** You are responsible for all setup, maintenance, and software configuration.
- **Platform as a Service (PaaS)**, like **Google's Vertex AI**, is a managed platform. It abstracts away the underlying hardware and provides a streamlined, high-level interface for training models.
  - **Pros:** Less setup, managed environment, often has built-in MLOps features.
  - **Cons:** Less control, can be more "magical" and harder to debug, may require code changes to fit the platform's API.

**Our Decision:** We chose **Google Compute Engine (GCE)** because our goal was to scale up an existing, complex parallel script with minimal code changes. The full control offered by a VM was the most direct path from a local laptop to a more powerful server.

---

## Part 2: The "What" — Selecting the Right Hardware

This was the most critical part of our deliberation, involving several discoveries and a key change in strategy.

### Identifying the Bottleneck: CPU-Bound Actors

Our script has two main components:

1.  **Actors:** Many parallel processes that run game simulations (MCTS) to generate training data. This is computationally intensive but doesn't require a GPU.
2.  **Learner:** A single process that trains the neural network on the data from the actors. This can benefit from a GPU.

The key realization was that the **actors were the bottleneck**. Generating data was much slower than training on it. Therefore, our primary goal became maximizing the performance of our many CPU-bound actor processes.

### The GPU Rabbit Hole vs. The CPU-Optimized Path

- **Initial Idea:** Use a general-purpose `N1` VM with an entry-level NVIDIA T4 GPU. This seems logical for deep learning.
- **Problem:** This would accelerate the learner, but the actors would still be running on older, less efficient CPUs, leaving the main bottleneck unresolved.
- **New Idea:** Use a **Compute-Optimized (C-series)** VM, which has much more powerful CPUs. We first considered the `C2` series.
- **Critical Insight:** After consulting the documentation, we discovered that **C-series VMs do not support attaching GPUs**. This forced a choice: a powerful CPU or a GPU. Given our bottleneck, the powerful CPU was the clear winner.

### The Power of Modern CPUs: AMX and PyTorch

Modern Intel Xeon CPUs (found in `C3` and `C4` series VMs) include **Advanced Matrix Extensions (AMX)**. AMX is specialized hardware that dramatically accelerates matrix multiplication operations—the core of deep learning—using lower-precision `bfloat16` numbers.

PyTorch has native support for AMX. By wrapping our model's forward pass in `torch.autocast`, we can tell PyTorch to automatically leverage this hardware, potentially making a modern CPU outperform an older CPU with an entry-level GPU for our specific workload.

**Final Hardware Choice:** A **`C4` series VM**. The `C4` series features Intel's 5th Gen "Emerald Rapids" Xeon processors, which are even newer than the "Sapphire Rapids" in the `C3` series. This gives us the best possible CPU performance. We chose a `c4-highcpu-22` instance, which provides 22 vCPUs (11 physical cores) and 44GB of RAM.

---

## Part 3: The "How" — Setting Up the VM

Here are the step-by-step commands to get the machine ready.

1.  **Create the VM Instance (from your local machine):**

    ```bash
    gcloud compute instances create mcts-vm \
        --zone=us-central1-a \
        --machine-type=c4-highcpu-22 \
        --image-family=debian-12 \
        --image-project=debian-cloud \
        --boot-disk-size=100GB \
        --maintenance-policy=TERMINATE --restart-on-failure
    ```

    _Note: A 100GB boot disk prevents storage issues during installation._

2.  **Connect via SSH (from your local machine):**

    ```bash
    gcloud compute ssh mcts-vm --zone=us-central1-a
    ```

3.  **Install Essential Packages (inside the VM):**
    A fresh Debian image needs `git` and other build tools.

    ```bash
    sudo apt-get update && sudo apt-get install -y git wget build-essential
    ```

4.  **Install and Configure Miniconda (inside the VM):**

    ```bash
    # Download and run the installer
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    ```

    _Accept all defaults during the interactive script. When it asks to initialize conda, say `yes`._

    **Close and re-open the SSH session** or run `source ~/.bashrc` for the changes to take effect. You should see `(base)` at your terminal prompt.

5.  **Grant GitHub Access via Deploy Key:**
    To securely clone a private repository, create an SSH key on the VM and add its public part to GitHub.

    ```bash
    # Create a new SSH key (press Enter three times for defaults)
    ssh-keygen -t ed25519

    # Display the public key to copy it
    cat ~/.ssh/id_ed25519.pub
    ```

    Copy the entire output. In your GitHub repo, go to `Settings > Deploy Keys > Add deploy key`, paste the key, give it a name, and **do not** grant write access.

6.  **Clone the Project:**
    ```bash
    git clone git@github.com:<your-username>/<your-repo-name>.git
    cd <your-repo-name>/
    ```

---

## Part 4: The Code — Preparing for the Cloud

Two small but critical code changes were needed for stability and performance.

### 1. Performance: `torch.autocast`

To unlock the power of AMX on the CPU (or Tensor Cores on a GPU), we wrap the model's forward pass in a `torch.autocast` context manager. This allows PyTorch to automatically run operations in lower-precision `bfloat16` (on CPU) or `float16` (on CUDA), which is much faster.

```python
// In the learner's training step:
with torch.autocast(device_type=self.config.learning_device.type):
    model_outputs = self.model.model(states)
    loss, metrics = self.config.training_adapter.compute_loss(model_outputs, targets, extra_data)

// In the actor's prediction function:
with torch.autocast(device_type=self.device.type):
    return self.model(observation)
```

This change makes the code portable and performant across different hardware.

### 2. Stability: Multiprocessing with `spawn`

On Linux, Python's `multiprocessing` defaults to the `fork` start method, which can cause deadlocks with complex libraries that manage their own threadpools, like PyTorch. The `spawn` method is safer and more robust as it creates a clean new process.

We force this behavior at the start of the main `Trainer` class initialization.

```python
// In the Trainer's __init__ method:
multiprocessing.set_start_method('spawn', force=True)
```

---

## Part 5: The "Gotchas" — Troubleshooting and Tuning

These are the real-world problems we hit and how we solved them.

### Problem: "No space left on device" during install

- **Cause:** A standard `pip install torch` tries to download the massive CUDA-enabled version of PyTorch (~2GB+), which can exhaust the default disk space.
- **Solution:** Manually install the **CPU-only** version of PyTorch _before_ running the main installation script. This version is much smaller.

  ```bash
  # Create the conda environment first
  conda create -n mcts-playground python=3.10 -y

  # Activate it
  conda activate mcts-playground

  # Install CPU-only PyTorch
  pip install torch --index-url https://download.pytorch.org/whl/cpu

  # Now run the rest of your installation (e.g., from requirements.txt)
  # pip install -r requirements.txt
  ```

### Problem: Actors Are Hanging and Making No Progress

- **Cause:** **Thread Oversubscription**. By default, PyTorch and other libraries (like NumPy linked with MKL/OpenMP) will try to use all available CPU cores for their own internal multithreading. When you also use Python's `multiprocessing` to create many actor processes, these two systems fight for resources, leading to a deadlock where each process is waiting on the others.
- **Solution:** **Force each actor process to be single-threaded**. This tells PyTorch to stay within its own process and not spawn extra threads, resolving the conflict.
  ```python
  # Add this at the VERY beginning of the actor_worker function
  torch.set_num_threads(1)
  ```

### Performance Tuning 1: vCPUs vs. Physical Cores

- **Concept:** Cloud providers advertise **vCPUs**, which are typically hardware threads (via Intel's Hyper-Threading). A single physical CPU core can have two threads (vCPUs).
- **Insight:** For heavily computational tasks like our actors, performance scales best with the number of **physical cores**, not vCPUs. The two threads on a single core share resources and don't provide a 2x speedup.
- **Action:** Set `num_actors` to the number of physical cores. For a `c4-highcpu-22` (22 vCPUs), this is **11 physical cores**.

### Performance Tuning 2: The "Starving Learner" Problem

- **Cause:** If we set `num_actors` to the total number of physical cores (e.g., 11), the learner process (which runs alongside the actors) has no dedicated CPU resources. It has to fight with the 11 actors for CPU time, becoming a new bottleneck.
- **Solution:** The **"CPU Headroom"** strategy. Reserve one physical core for the learner and other system processes.
- **Action:** Set `num_actors = physical_cores - 1`. On our 11-core machine, this means setting `num_actors = 10`. This leaves one core free to ensure the learner can process incoming data and run training steps efficiently.

---

## Part 6: The Workflow — Running Long-Term Experiments

If you close your SSH connection, your running script will be killed. To run experiments for hours or days, you need a persistent terminal session.

### Using `tmux` for Persistent Sessions

`tmux` is a terminal multiplexer. It creates sessions on the server that you can detach from and re-attach to later, even after closing your local terminal.

- **Start a new named session (inside the VM):**
  ```bash
  tmux new -s training-run
  ```
- **Run your script inside the `tmux` session:**
  ```bash
  conda activate mcts-playground
  python your_script.py
  ```
- **Detach from the session:** Press `Ctrl+B`, then `D`. The session and your script are still running in the background. You can now safely close the SSH connection.
- **Re-attach to the session later:**

  ```bash
  # SSH back into the VM
  gcloud compute ssh mcts-vm --zone=us-central1-a

  # List running sessions
  tmux ls

  # Attach to your session by name
  tmux attach -t training-run
  ```

---

## Part 7: Cost Management

**This is the most important step.** Cloud VMs bill for every minute they are running.

1.  **To STOP the VM (and stop billing for compute time):**
    Run this from your **local machine's** terminal. Your files and disk are preserved.

    ```bash
    gcloud compute instances stop mcts-vm --zone=us-central1-a
    ```

2.  **To RESTART the VM:**
    Run this from your local machine.
    ```bash
    gcloud compute instances start mcts-vm --zone=us-central1-a
    ```
    You can then SSH back in and re-attach to your `tmux` session to check on your experiment.
