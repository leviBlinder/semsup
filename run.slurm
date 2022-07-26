#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=comp

# Remove one # to uncommment
#SBATCH --output=%x-%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 2 #tasks
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=0-6:00:00    # Run for 6 hours
#SBATCH --gres=gpu:2

# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=NONE
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=nobody@cs.princeton.edu


# Define and create a unique scratch directory for this job
tag=aug14_curriculum;
OUT_DIRECTORY='output/'${tag}
mkdir ${OUT_DIRECTORY};

# Submit jobs.

bash run_cifar_semsup.sh

wait; #Make sure to wait till all the runs have completed.

# Finish the script
exit 0