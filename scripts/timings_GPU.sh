#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --partition=cuda
#SBATCH --mail-user=smerlak@mis.mpg.de
#SBATCH --mail-type=ALL

julia timings_GPU.jl > timings_GPU.log
