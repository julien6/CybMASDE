#!/bin/bash

wget https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh --no-proxy;
chmod +x install.sh
./install.sh;

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

source ~/.bashrc
nvm install node;
rm -rf install.sh
npm install -g @angular/cli
npm i
