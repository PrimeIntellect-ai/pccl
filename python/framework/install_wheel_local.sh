#!/usr/bin/env bash

bash ./build_wheel.sh
pip3 install ./dist/*.whl --force-reinstall
