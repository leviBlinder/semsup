""" few-shot learning pretrain CIFAR100 datamodule
"""
from dataclasses import dataclass

from torch.utils.data import Subset
import torchvision
from torchvision.datasets.vision import StandardTransform
from sklearn.model_selection import train_test_split

from .base import CIFARDataArgs, CIFAR100DataModule


@dataclass
class CIFARFewShotPretrainDataArgs(CIFARDataArgs):
    """Data arguments for CIFAR FSL pretraining.
    Args:
        fewshot_names (tuple): classes not seen in pretraining
    """
    fewshot_names: tuple = (
        "motorcycle",
        "pine_tree",
        "bottle",
        "trout",
        "chair"
    )


    # fewshot_names: tuple = (
    #     "motorcycle",
    #     "pine_tree",
    #     "bottle",
    #     "trout",
    #     "chair"
    #     "butterfly",
    #     "chimpanzee",
    #     "orange",
    #     "leopard",
    #     "possum",
    # )



    def __post_init__(self):
        super().__post_init__()


class CIFARFewShotPretrainDM(CIFAR100DataModule):
    """class which iplements FSL pretraining on cifar"""

    def __init__(self, args: CIFARFewShotPretrainDataArgs, *margs, **kwargs):
        temp_classes = args.classes
        args.classes = tuple([x for x in temp_classes if x not in args.fewshot_names])
        args.train_classes = args.classes
        args.val_classes = args.classes
        super().__init__(args, *margs, **kwargs)

    def setup(self, stage=None):
        super().setup(stage)
        #sanity checks
        #print("train classes: ", self.args.train_classes)
        #print("val classes: ", self.args.val_classes)
        assert self.args.train_classes == self.args.val_classes, "Mismatched classes"
        #print("classes: ", self.args.classes)
        assert self.args.val_classes == self.args.classes, "Mismatched classes"
        class_ids = [self.all_classlabel.str2int(class_name) for class_name in self.args.classes]
        for idx in range(len(self.dataset["train"])):
            _, class_id = self.dataset["train"][idx]
            assert class_id in class_ids, f"In train dataset, {class_id} not a valid class name"
        for idx in range(len(self.dataset["val"])):
            _, class_id = self.dataset["val"][idx]
            assert class_id in class_ids, f"In val dataset, {class_id} not a valid class name"

if __name__ == "__main__":
    # unit tests
    data_args = CIFARFewShotPretrainDataArgs(
        label_tokenizer="prajjwal1/bert-small",
        train_label_json="../class_descs/cifarGPT_formatted.labels",
        cache_dir="../data_cache",
    )
    data_mod = CIFARFewShotPretrainDM(args=data_args)
    data_mod.prepare_data()
    data_mod.setup()
    for _ in data_mod.train_dataloader():
        pass
    for _ in data_mod.val_dataloader():
        pass
