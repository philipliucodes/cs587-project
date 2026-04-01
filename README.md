# Virtual environment set-up

ml conda
conda create -p ./env python=3.10
conda activate ./env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Experiment commands

Baseline (standard SGD)
python main.py --dataset=cifar10 --data-aug=0 --model=PreActResNet18 --method=0

Ordered SGD
python main.py --dataset=cifar10 --data-aug=0 --model=PreActResNet18 --method=1

the ones below here are performing terribly, need use hyperparameters or implement some kind of scheduling probably

KL-DRO/exponential
python main.py --dataset=cifar10 --data-aug=0 --model=PreActResNet18 --method=2

Z-score weighting
python main.py --dataset=cifar10 --data-aug=0 --model=PreActResNet18 --method=3

Focal weighting
python main.py --dataset=cifar10 --data-aug=0 --model=PreActResNet18 --method=5

## Refer to the scripts/ folder to use preset configs and commands for each experiment

## Standards and conventiosn for GitHub Versioning

Make a Branch for any changes you make, and make sure to pull from main before merging

[version type]: [what the version did]

- fix: Small change that has significance, not a bug fix but should be no more than a few lines
- feat: Significant change, when something major is added, like a new method
- wip: should only be done in branches, only do if you're logging off so that other people dont work off or merge a branch with a wip: tag at the end
- bug: small bugfix, one line changes that have little significance

lmao - philip
