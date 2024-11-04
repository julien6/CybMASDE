#!/bin/bash

wget https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh --no-proxy;
chmod +x install.sh
./install.sh;
source ~/.bashrc
nvm install node;
rm -rf install.sh
npm install -g @angular/cli
npm i

ng serve