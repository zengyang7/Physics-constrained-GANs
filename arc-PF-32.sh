#!/bin/bash

#PBS -l procs=1,gpus=1
#PBS -l walltime=00:10:00
#PBS -q p100_normal_q
#PBS -A mfulearn
#PBS -W group_list=newriver
#PBS -M yangzeng@vt.edu
#PBS -m bea

cd $PBS_O_WORKDIR

module purge
module load Anaconda/5.1.0
module load cuda/9.0.176
module load cudnn/7.1

python cWGANGP-Bernoulli-32.py
