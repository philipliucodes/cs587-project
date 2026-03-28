Virtual environment set-up:

ml conda
conda create -p ./env python=3.10
conda activate ./env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
