#!/bin/bash

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
fi

cat << 'EOT' > tmp_install.sh
#!/bin/bash

if [ ! -f "/applis/environments/conda.sh" ] && [ ! -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    echo -e "\n\nINSTALLATION OF MINICONDA\n"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh
fi

if [ -e "/applis/environments/conda.sh" ]; then
    source /applis/environments/conda.sh
else
    source ~/miniconda3/etc/profile.d/conda.sh
fi

conda --version
conda create -n marllib python=3.8 -y
conda activate marllib


rm -rf Miniconda3-latest-Linux-x86_64.sh
git clone https://github.com/Replicable-MARL/MARLlib.git
cd MARLlib
pip install --upgrade pip
pip install -r requirements.txt
pip install setuptools==65.5.0 pip==21
pip install wheel==0.38.0
pip install scikit-learn
pip install -r requirements.txt
pip install "gym>=0.20.0,<0.22.0"
pip install ray[tune]

pip install protobuf==3.20.*
python marllib/patch/add_patch.py -y

export PYTHONPATH="${PYTHONPATH}:./src"

pip install pettingzoo # ==1.23.1
pip install supersuit # ==3.9.0
pip install pygame==2.3.0
pip install numpy==1.23.5

pip install pettingzoo --upgrade
pip install supersuit --upgrade

conda install -c conda-forge libstdcxx-ng
pip install pyglet==1.5.11

pip install marllib==1.0.3
pip install numpy==1.23.5
pip install torch -U

# Obtenir le chemin absolu du script en cours d'exécution
declare script_path="$(dirname "$(readlink -f "$BASH_SOURCE")")"

cd \$script_path

pip install -r requirements.txt
EOT

# echo -e "\n\nINSTALLATION ON INTERFACE LOCAL MACHINE\n"
chmod +x tmp_install.sh
./tmp_install.sh;
rm -rf tmp_install.sh

# if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
#     echo -e "\n\nINSTALLATION ON COMPUTATIONAL REMOTE SERVER\n"
#     ssh soulej@bigfoot.ciment -t "cd /bettik/soulej ; rm -rf omarl_experiments ; mkdir omarl_experiments"
#     rsync -avxH tmp_install.sh oar_launch.sh train_test.py  soulej@bigfoot.ciment:/bettik/soulej/omarl_experiments/
#     rm -rf tmp_install.sh
#     ssh soulej@bigfoot.ciment -t "cd /bettik/soulej/omarl_experiments ; ./tmp_install.sh ; rm -rf tmp_install.sh ; echo -e \"\n\nINSTALLATION FINISHED!\n\""
# fi

# if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
#     echo -e "\n\nINSTALLATION ON COMPUTATIONAL REMOTE SERVER\n"
#     ssh soulej@bigfoot.ciment -t "cd /bettik/soulej ; rm -rf omarl_experiments ; git clone https://github.com/julien6/omarl_experiments.git ; cd omarl_experiments ; git checkout test ; cd prahom_wrapper/tests/test"
#     rsync -avxH tmp_install.sh soulej@bigfoot.ciment:/bettik/soulej/omarl_experiments/prahom_wrapper/tests/test
#     rm -rf tmp_install.sh
#     ssh soulej@bigfoot.ciment -t "cd /bettik/soulej/omarl_experiments/prahom_wrapper/tests/test ; ./tmp_install.sh ; rm -rf tmp_install.sh ; echo -e \"\n\nINSTALLATION FINISHED!\n\""
# fi
