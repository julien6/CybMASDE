#!/usr/bin/env bash

git config --global http.postBuffer 524288000
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999
git config --global core.compression 0

# -----------------------------------
# Vérification Python 3.8
# -----------------------------------
if command -v python3.8 >/dev/null 2>&1; then
    echo "✅ Python 3.8 was found."
else
    echo "❌ Python 3.8 is not installed. Please install it (ex: sudo apt install python3.8 python3.8-venv)."
    exit 1
fi

echo "🛠️  Creating virtual Python 3.8 environment..."
python3.8 -m venv env

echo "Activation de l'environnement virtuel..."
source env/bin/activate

pip install --upgrade pip setuptools

git clone https://github.com/julien6/PettingZoo.git

cd PettingZoo

pip install -e .

cd ..

# Install OvercookedAI
git clone https://github.com/julien6/overcooked_ai.git

cd overcooked_ai

pip install -e .

cd ..

# Install dependencies
pip install -r requirements.txt
