#!/bin/sh
#SBATCH --output=jobs/compare-verifiers-%j.out
eval "$(conda shell.bash hook)"
conda activate d3
python /scratch/users/petez/DescribeDistributionalDifferences/system_v05-14-22/compare_verifiers.py
