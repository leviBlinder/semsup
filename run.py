import yaml
from argparse import ArgumentParser
from jsonargparse import ActionConfigFile
from jsonargparse import ArgumentParser as JSONArgumentParser
from jsonargparse.typing import Path_fr
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import semsup.models
import semsup.data


def parse_arg_with_default(default_config, args, key, required = False):
    default = default_config[key] if key in default_config else None
    ret = getattr(args, key) if getattr(args, key) else default
    if required and ret is None:
        raise Exception(f"{key} is a required argument")
    return ret
    

if __name__ == "__main__":
    cls_parser = ArgumentParser()
    cls_parser.add_argument("--default_config", type=str, help="path to the default config")
    cls_parser.add_argument("--config", type=str, help="the configuration file to use")
    cls_parser.add_argument("--ModelCls", type=str, help="model class to use")
    cls_parser.add_argument("--DataCls", type=str, help="data class to use")
    cls_args, _ = cls_parser.parse_known_args()

    with open(cls_args.config) as f:
        cls_config = yaml.safe_load(f)

    ModelCls = parse_arg_with_default(cls_config, cls_args, "ModelCls", required=True)
    ModelCls = getattr(semsup.models, ModelCls)
    
    DataCls = parse_arg_with_default(cls_config, cls_args, "DataCls", required=True)
    DataCls = getattr(semsup.data, DataCls)

    parser = JSONArgumentParser(default_config_files=[cls_args.default_config])
    parser.add_argument("--default_config", type=Path_fr, help="path to the default config")
    parser.add_argument("--config", action=ActionConfigFile, help="the configuration file to use")
    parser.add_argument("--ModelCls", type=str, help="model class to use")
    parser.add_argument("--DataCls", type=str, help="data class to use")
    parser.add_argument("--name", type=str, help="name of the experiment to run")
    parser.add_argument("--seed", type=int, help="int to seed everything with")
    parser.add_argument("--load.checkpoint", type=Path_fr, help="path to load model checkpoint")
    parser.add_argument("--load.hyperparams", action="store_true", help="use the hyperparams from the checkpoint")
    parser.add_argument("--validate", action="store_true", help="do validation")
    parser.add_argument("--train", action="store_true", help="do training")
    parser.add_argument("--name_suffix", type=str, help="suffix to add to the logger and ckpt name")
    parser.add_argument("--logger.group", type=str, help="name of the group to log the runs to")

    parser.add_class_arguments(ModelCls, "model")
    parser.add_class_arguments(DataCls, "data")
    parser.add_class_arguments(pl.Trainer, "trainer")
    parser.add_class_arguments(WandbLogger, "logger")
    parser.add_class_arguments(ModelCheckpoint, "checkpoint")

    args = parser.parse_args()
    
    pl.seed_everything(args.seed)
    if args.name_suffix:
        args.name = args.name + "_" + args.name_suffix
    args.logger.name = args.name if args.name else None
    #args.checkpoint.filename = args.name if args.name else None
    
    cfg = parser.instantiate_classes(args)
    cfg.checkpoint.CHECKPOINT_NAME_LAST = "last_" + args.name 
    
    if cfg.load.checkpoint:
        print("Checkpoint Path: ", cfg.load.checkpoint)
        cfg.model = (
            ModelCls.load_from_checkpoint(cfg.load.checkpoint)
            if cfg.load.hyperparams else
            ModelCls.load_from_checkpoint(cfg.load.checkpoint, args=cfg.model.args)
        )
        print("Checkpoint Model Loaded")
    if args.trainer.logger:
        cfg.logger.watch(cfg.model) #log="all", log_freq=5000)
        cfg.logger.experiment.config.update(args)

    # modify the trainer, since it depends on instantiated classes like loggers and callbacks
    args.trainer.logger = cfg.logger if args.trainer.logger else False
    args.trainer.callbacks = [cfg.checkpoint] if args.trainer.checkpoint_callback else []
    cfg.trainer = pl.Trainer(**args.trainer.as_dict())

    if cfg.validate:
        cfg.trainer.validate(cfg.model, cfg.data)

   
    # try:
    if cfg.train:
        cfg.trainer.fit(cfg.model, cfg.data)
    # except Exception as ex:
        # print(cfg.data)
        # print("RUN: train transform: ", cfg.data.dataset["train"].dataset.transform)
        # print("RUN: val transform: ", cfg.data.dataset["val"].dataset.transform)
        # print("RUN: train target_transform: ", cfg.data.dataset["train"].dataset.target_transform)
        # print("RUN: val target_transform: ", cfg.data.dataset["val"].dataset.target_transform)
        # print("RUN: train transforms: ", cfg.data.dataset["train"].dataset.transforms)
        # print("RUN: val transforms: ", cfg.data.dataset["val"].dataset.transforms)
        # print("RUN: train data: ", cfg.data.dataset["train"])
        # print("RUN: train dataset el1: ", cfg.data.dataset["train"][0])
        # print("RUN: train dataset el1 transform: ", cfg.data.dataset["train"].dataset.transform(cfg.data.dataset["train"][0][0]))
        # print("RUN: train dataset el1 targ transform: ", cfg.data.dataset["train"].dataset.target_transform(cfg.data.dataset["train"][0][1]))
        # print("RUN: train dataset el1 transforms: ", cfg.data.dataset["train"].dataset.transforms(*cfg.data.dataset["train"][0]))
        # print("RUN: train dataset el1-2: ", cfg.data.dataset["train"].dataset[0])
        # raise ex
    
    # print(cfg.data)
    # print("RUN: train transform: ", cfg.data.dataset["train"].dataset.transform)
    # print("RUN: val transform: ", cfg.data.dataset["val"].dataset.transform)
    # print("RUN: train target_transform: ", cfg.data.dataset["train"].dataset.target_transform)
    # print("RUN: val target_transform: ", cfg.data.dataset["val"].dataset.target_transform)
    # print("RUN: train transforms: ", cfg.data.dataset["train"].dataset.transforms)
    # print("RUN: val transforms: ", cfg.data.dataset["val"].dataset.transforms)
    # print("RUN: train data: ", cfg.data.dataset["train"])
    