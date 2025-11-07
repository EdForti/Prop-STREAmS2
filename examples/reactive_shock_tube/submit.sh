#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH --time 00:30:00      # format: HH:MM:SS
#SBATCH -N 1                 # number of nodes N
#SBATCH --ntasks-per-node=1  # 12 tasks out of 32, 24*N=48 
#SBATCH --gpus-per-node=1    # 2 gpus out of 4, 4*N=8 gpus
#SBATCH -A CNHPC_2021201
#SBATCH --qos=boost_qos_dbg
#SBATCH --job-name=RSTTest
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --error=job.err

module purge
module load profile/global ;
module load cmake/3.24.3 ;
module load gcc/11.3.0 ;
module load cuda/11.8 ;
module load python/3.10.8--gcc--11.3.0 ;
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8 ;
module load nvhpc/25.3 ;
module load hpcx-mpi/2.19 ;

srun /leonardo/home/userexternal/eforti00/streams_2_reactive/code_13sp/streams_2.exe > data.dat
