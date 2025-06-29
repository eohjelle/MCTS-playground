# Guide: Setting Up Google Cloud for CPU-Based Training

This document outlines the end-to-end process for setting up a powerful, CPU-optimized Google Cloud VM to run the MCTS training script.

---

### Phase 1: Create and Connect to the VM

We use a CPU-optimized `C4` series VM, which features modern Intel Xeon processors with AMX hardware for accelerating deep learning tasks on the CPU.

1.  **Create the VM Instance (from your local machine):**
    This command provisions a `c4-highcpu-22` machine with 22 vCPUs (11 physical cores) and 44GB of RAM in the `us-central1-a` zone.

    ```bash
    gcloud compute instances create mcts-vm \
        --zone=us-central1-a \
        --machine-type=c4-highcpu-22 \
        --image-family=debian-12 \
        --image-project=debian-cloud \
        --boot-disk-size=100GB \
        --maintenance-policy=TERMINATE --restart-on-failure
    ```

    _Note: We are using a standard Debian 12 image and will set it up from scratch._

2.  **Connect via SSH (from your local machine):**

    ```bash
    gcloud compute ssh mcts-vm --zone=us-central1-a
    ```

---

### Phase 2: Configure the VM Environment

These commands are run inside the VM's terminal after you have connected via SSH.

1.  **Install Essential Packages:**
    A standard Debian image needs `git` and other build tools.

    ```bash
    sudo apt-get update && sudo apt-get install -y git wget build-essential
    ```

2.  **Install Miniconda (for Conda environments):**

    ```bash
    # Download the installer
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

    # Run the installer, accepting all defaults
    bash Miniconda3-latest-Linux-x86_64.sh
    ```

    _During the interactive script, press Enter, type `yes` to agree to the license, press Enter again to confirm the location, and type `yes` to initialize conda._

3.  **Activate Conda for the Session:**
    You must close and reopen the SSH session, or run the following command for the changes to take effect:
    ```bash
    source ~/.bashrc
    ```
    You should now see `(base)` at your terminal prompt.

---

### Phase 3: Set Up Project & Dependencies

1.  **Grant GitHub Access (for private repos):**
    To clone the private repository, create an SSH key on the VM and add it as a "Deploy Key" on GitHub.

    ```bash
    # Create a new SSH key
    ssh-keygen -t ed25519
    ```

    (Press Enter three times to accept defaults with no passphrase).

    ```bash
    # Display the public key to copy it
    cat ~/.ssh/id_ed25519.pub
    ```

    Copy the output and add it as a new **Deploy Key** in your GitHub repository's settings (`Settings > Deploy Keys > Add deploy key`). Do not give it write access.

2.  **Clone the Project:**

    ```bash
    git clone git@github.com:<your-username>/<your-repo-name>.git
    cd <your-repo-name>/
    ```

3.  **Install Dependencies (The CPU-Only Method):**
    The `install.sh` script will fail due to a "No space left on device" error because it tries to download the massive CUDA-enabled version of PyTorch. To solve this without modifying the script:

    - **First, create the environment:**
      ```bash
      conda create -n mcts-playground python -y
      ```
    - **Next, install the CPU-only PyTorch:**
      ```bash
      conda run -n mcts-playground pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
      ```
    - **Finally, run the rest of the original `install.sh` script**, which will now skip the PyTorch installation and proceed with `wandb`, `pytest`, and `open_spiel`.

---

### Phase 4: Run the Experiment

1.  **Activate the Environment:**

    ```bash
    conda activate mcts-playground
    ```

2.  **Log in to Weights & Biases:**

    ```bash
    wandb login
    ```

    (Paste your API key when prompted).

3.  **Run the Training Script:**
    Launch your script. For best performance, set `num_actors` in your configuration to match the number of **physical cores** (11 on a `c4-highcpu-22` instance), not the number of vCPUs.

---

### Phase 5: Manage VM and Control Costs

**This is the most important step for cost management.**

1.  **To Stop the VM (and stop billing for compute time):**
    Run this from your **local machine's** terminal:

    ```bash
    gcloud compute instances stop mcts-vm --zone=us-central1-a
    ```

2.  **To Restart the VM:**
    Your files and setup are preserved. Run this from your local machine:
    ```bash
    gcloud compute instances start mcts-vm --zone=us-central1-a
    ```
    You can then SSH back in and resume your work.
