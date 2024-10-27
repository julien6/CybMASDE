#!/bin/bash

python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
# echo 'Please type: "source venv/bin/activate"'
python ./src/api_server/server.py