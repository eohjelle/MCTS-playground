# Create conda environment
conda create -n mcts-playground python -y

# Install required python packages
conda run -n mcts-playground pip install -e .

# Install open_spiel directly from source, pip installation does not always work
git clone https://github.com/google-deepmind/open_spiel.git
cd open_spiel
conda run -n mcts-playground ./install.sh
conda run -n mcts-playground pip install .
cd ..
rm -rf open_spiel