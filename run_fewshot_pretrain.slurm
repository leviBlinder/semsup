#!/bin/bash

#SBATCH --job-name cifar
#SBATCH --output=slurm/%x-%j

#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -t 3:00:00
#SBATCH -x node901,node902,node000,node001,node010,node011,node016,node916

bash run_cifar_semsup_fewshot_pretrain.sh --data.args.overwrite_label_cache true