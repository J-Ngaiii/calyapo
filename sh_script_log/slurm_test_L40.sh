#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=fc_hartmanl2
#
# Partition:
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task:
# Eight times the number for L40 in savio4_gpu
#SBATCH --cpus-per-task=8
#
#Number and type of GPUs
#SBATCH --gres=gpu:1 # requesting L40 explicitly does not work
#SBATCH --qos=savio_lowprio

# Wall clock limit:
#SBATCH --time=00:10:00

sleep 60h