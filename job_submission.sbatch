#!/bin/bash

#SBATCH --job-name=job_submission

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:1080ti:1

#SBATCH --partition=test
#SBATCH --time=11:11
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=/home/stud143/RL_project/logs/job_submission.err
#SBATCH --output=/home/stud143/RL_project/logs/job_submission.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=paulina.stark@student.uni-tuebingen.de

# Add the directory containing 'project' to PYTHONPATH
export PYTHONPATH=/home/stud143/RL_project:$PYTHONPATH

srun -N1 -n1 -c $SLURM_CPUS_PER_TASK --gres=gpu:1080ti:1 \
    singularity exec --nv /home/stud143/RL_project/container.sif /venv/bin/python /home/stud143/RL_project/project/scripts/train.py
echo "DONE!"
