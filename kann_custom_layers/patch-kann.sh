#!/usr/bin/env bash

URL="/nfs/users/qmuller/modelsKalray/.wheels/"
PYWHEEL="kann-5.1.0-cp310-cp310-linux_x86_64.whl"

pip uninstall kann
pip install "$URL/$PYWHEEL"
if [[ $(kann --version) != "kann, version 5.1.0-extended" ]]; then
    exit 1
fi