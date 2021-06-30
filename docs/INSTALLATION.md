FISH Installation Guide 
===========
Follow the instructions to install anaconda on your machine (download the distribution that comes with python 3):
https://www.anaconda.com/distribution/

After setting up anaconda, first downgrade/upgrade the setuptools to avoid confliction
```shell
sudo apt-get install openslide-tools
```

Next, clone our repo and use the environment configuration file located in **docs/fish.yaml** to create a conda environment:
```shell
git clone https://github.com/mahmoodlab/FISH.git
cd ./FISH
conda env create -n fish -f docs/fish.yaml
```

Activate the environment and install openslide-python:
```shell
conda activate fish
pip install openslide-python==1.1.1
```
