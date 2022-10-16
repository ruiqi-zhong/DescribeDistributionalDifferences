#!/bin/sh
#SBATCH --output=jobs/test-iterative-%j.out
eval "$(conda shell.bash hook)"
conda activate d3
python /scratch/users/petez/DescribeDistributionalDifferences/system_v05-14-22/test_iterative.py
