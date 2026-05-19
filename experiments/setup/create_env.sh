#!/bin/bash
# Create conda environment for Functional Atlas experiments
# Run on HPC3: bash create_env.sh

CONDA_BASE="/data/user/mzhang630/miniconda3"
ENV_NAME="funcatlas"

source ${CONDA_BASE}/etc/profile.d/conda.sh

conda create -n ${ENV_NAME} python=3.11 -y
conda activate ${ENV_NAME}

pip install torch==2.3.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.0 accelerate datasets tokenizers sentencepiece
pip install scikit-learn scipy numpy pandas matplotlib seaborn
pip install lm-eval==0.4.3
pip install networkx  # for graph-theoretic analysis
pip install huggingface_hub

# Data directory
mkdir -p /data/user/mzhang630/data/nature_exp/{models,activations,modules,results,stimuli}

echo "Environment ${ENV_NAME} created successfully."
echo "Activate with: conda activate ${ENV_NAME}"
