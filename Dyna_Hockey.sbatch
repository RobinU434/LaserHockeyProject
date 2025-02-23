#!/bin/bash

#SBATCH --job-name=Dyna-Pendulum

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:1080ti:1

#SBATCH --partition=day
#SBATCH --time=10:00:00
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=/home/stud466/LaserHockeyProject/logs/Dyna_Pendulum.err
#SBATCH --output=/home/stud466/LaserHockeyProject/logs/Dyna_Pendulum.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=robin.uhrich@student.uni-tuebingen.de
#SBATCH --mail-user=paulina.stark@student.uni-tuebingen.de

# Add the directory containing 'project' to PYTHONPATH
export PYTHONPATH=/home/stud466/LaserHockeyProject:$PYTHONPATH

# Run the correct command
srun -N1 -n1 -c $SLURM_CPUS_PER_TASK --gres=gpu:1080ti:1 \
        singularity exec --nv /home/stud466/LaserHockeyProject/container.sif \
        /venv/bin/python /home/stud466/LaserHockeyProject/project/__main__.py train-dyna-hockey --force --device cuda:0

echo "DONE!"
