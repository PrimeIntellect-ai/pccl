#!/usr/bin/env bash

rm -rf ./build
rm -rf ./dist
rm -rf ./pccl.egg-info
pip3 wheel -w dist .
