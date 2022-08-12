#!/bin/bash

SEED=1
RUN_SCRIPT="../run.py"
CONFIG_FOLDER="./cifar_configs"

commands=(
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/cifar_fewshot_pretrain.yaml" 
)
ARGS="--train --default_config $CONFIG_FOLDER/cifar_default.yaml --seed $SEED --name_suffix s$SEED"

for CMD in "${commands[@]}"; do
    $CMD $ARGS $@
    if [ "$?" -ne "0" ]; then
    exit 1
    fi
done