#!/bin/bash
set -e

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
conda create -n dev python=3.11 -y
conda activate dev
echo "conda activate dev" >> ~/.bashrc
git clone https://github.com/awisniewski21/ShengChengZi.git ~/ShengChengZi
cd ~/ShengChengZi
pip install -r requirements.txt
pip install -e .
apt update
apt install -y screen