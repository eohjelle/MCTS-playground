# Create virtual environment and install required python packages
uv sync

# Install open_spiel directly from source, since pip installation does not always work
git clone https://github.com/google-deepmind/open_spiel.git
cd open_spiel
./install.sh
cd ..
uv pip install ./open_spiel
rm -rf open_spiel
