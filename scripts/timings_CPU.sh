#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --partition=epyc
#SBATCH --mail-user=smerlak@mis.mpg.de
#SBATCH --mail-type=ALL


julia timings_CPU.jl > timings_CPU.log
