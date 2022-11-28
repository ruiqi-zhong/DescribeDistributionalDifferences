#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate env

python3 get_rep.py -i $index
