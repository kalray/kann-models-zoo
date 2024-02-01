#!/usr/bin/env bash

URL="/nfs/users/qmuller/modelsKalray/.wheels/"
PYWHEEL="kann-5.0.1-cp310-cp310-linux_x86_64.whl"

pip uninstall kann
pip install "$URL/$PYWHEEL"
if [[ $(kann --version) != "kann, version 5.0.0" ]]; then
    exit 1
fi