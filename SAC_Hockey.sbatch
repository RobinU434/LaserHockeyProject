#!/bin/bash

#SBATCH --job-name=SAC-Pendulum

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:1080ti:1

#SBATCH --partition=day
#SBATCH --time=04:00:00
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=/home/stud143/projects/LaserHockeyProject/logs/SAC_hockey.err
#SBATCH --output=/home/stud143/projects/LaserHockeyProject/logs/SAC_hockey.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=paulina.stark@student.uni-tuebingen.de
#SBATCH --mail-user=robin.uhrich@student.uni-tuebingen.de

# Add the directory containing 'project' to PYTHONPATH
export PYTHONPATH=/home/stud143/projects/LaserHockeyProject:$PYTHONPATH

# Run the correct command
srun -N1 -n1 -c $SLURM_CPUS_PER_TASK --gres=gpu:1080ti:1 \
	singularity exec --nv /home/stud143/projects/LaserHockeyProject/container.sif \
	/venv/bin/python /home/stud143/projects/LaserHockeyProject/project/__main__.py train-sac-hockey --force

echo "DONE!"
