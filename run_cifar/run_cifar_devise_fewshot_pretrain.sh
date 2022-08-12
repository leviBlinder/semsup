#!/bin/bash

SEED=1
RUN_SCRIPT="../run.py"
CONFIG_FOLDER="./cifar_configs"

commands=(
    "python $RUN_SCRIPT --config $CONFIG_FOLDER/cifar_fewshot_clsname_pretrain.yaml"
)

ARGS="--train --default_config $CONFIG_FOLDER/cifar_default.yaml --ModelCls DEVISEVisBaseline --data.args.setup_glove_embeddings true --seed $SEED --name_suffix devise_s$SEED"

export GENSIM_DATA_DIR=/n/fs/nlp-lblinder/semsup
for CMD in "${commands[@]}"; do
    $CMD $ARGS $@
    if [ "$?" -ne "0" ]; then
    exit 1
    fi
done