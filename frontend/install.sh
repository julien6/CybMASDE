#!/bin/bash

curl -o nvm_install.sh https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh
chmod +x nvm_install.sh
./nvm_install.sh

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"                   # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion" # This loads nvm bash_completion

source ~/.bashrc
nvm install node
rm -rf nvm_install.sh
npm install -g @angular/cli
npm i
