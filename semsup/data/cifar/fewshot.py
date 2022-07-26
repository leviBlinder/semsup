""" few-shot learning CIFAR100 datamodule
"""
from dataclasses import dataclass

from torch.utils.data import Subset
import torchvision
from torchvision.datasets.vision import StandardTransform
from sklearn.model_selection import train_test_split

from .base import CIFARDataArgs, CIFAR100DataModule


@dataclass
class CIFARFewShotDataArgs(CIFARDataArgs):
    """Data arguments for CIFAR FSL.
    Args:
        fewshot_names (tuple): classes not seen in pretraining
        few_shot_k (int): Sets number of heldout images to add to train data per heldout class.
    """
    fewshot_names: tuple = (
        "motorcycle",
        "pine_tree",
        "bottle",
        "trout",
        "chair",
        "butterfly",
        "chimpanzee",
        "orange",
        "leopard",
        "possum",
    )
    few_shot_k: int = 0 # Number of samples from heldout classes to include in training, should be overidden



    def __post_init__(self):
        super().__post_init__()
        self.train_size = int(self.few_shot_k * len(self.fewshot_names))
        self.val_size = None
        print("fewshot_k: ", self.few_shot_k)
        print("train_size: ", self.train_size)


class CIFARFewShotDM(CIFAR100DataModule):
    """class which iplements few-shot pretraining on cifar"""

    def __init__(self, args: CIFARFewShotDataArgs, *margs, **kwargs):
        args.classes = args.fewshot_names
        args.train_classes = args.classes
        args.val_classes = args.classes
        super().__init__(args, *margs, **kwargs)

    def setup(self, stage=None):
        super().setup(stage)
        #sanity checks
        print("train classes: ", self.args.train_classes)
        print("val classes: ", self.args.val_classes)
        #print("classes: ", self.args.classes)
        print("train length: ", len(self.dataset["train"]))
        print("val length: ", len(self.dataset["val"]))
        #print("test length: ", len(self.dataset["train"]))
        class_ids = [self.all_classlabel.str2int(class_name) for class_name in self.args.classes]
        assert self.args.train_classes == self.args.val_classes == self.args.classes == self.args.fewshot_names, "Mismatched classes"
        for idx in range(len(self.dataset["train"])):
            _, class_id = self.dataset["train"][idx]
            assert class_id in class_ids, f"In train dataset, {class_id} not a valid class id"
        for idx in range(len(self.dataset["val"])):
            _, class_id = self.dataset["val"][idx]
            assert class_id in class_ids, f"In val dataset, {class_id} not a valid class id"
        if self.args.run_test:
            for idx in range(len(self.dataset["test"])):
                _, class_id = self.dataset["test"][idx]
                assert class_id in class_ids, f"In test dataset, {class_id} not a valid class id"


if __name__ == "__main__":
    # unit tests
    data_args = CIFARFewShotDataArgs(
        label_tokenizer="prajjwal1/bert-small",
        train_label_json="../class_descs/cifarGPT_formatted.labels",
        cache_dir="../data_cache",
    )
    data_mod = CIFARFewShotDM(args=data_args)
    data_mod.prepare_data()
    data_mod.setup()
    for _ in data_mod.train_dataloader():
        pass
    for _ in data_mod.val_dataloader():
        pass
