#!/bin/bash
#Below is the queue type, use ai for GPU usages
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=32:mem=128G
#PBS -l walltime=48:00:00
#PBS -P Personal
#PBS -N flow-matching-minibatch-checkerboard
#PBS -m abe
#PBS -M tuanbinhs@gmail.com
#PBS -o $HOME/pbs_logs/$PBS_JOBNAME-$PBS_JOBID-out.txt
#PBS -e $HOME/pbs_logs/$PBS_JOBNAME-$PBS_JOBID-err.txt

source $HOME/miniconda3/bin/activate ottest
cd $HOME/scratch/work/lfm

set -x
set -e

python example_gaussian_checkerboard.py 
