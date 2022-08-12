""" few-shot learning CIFAR100 datamodule
"""
from dataclasses import dataclass

from torch.utils.data import Subset
import torchvision
from torchvision.datasets.vision import StandardTransform
from sklearn.model_selection import train_test_split

from .base import CIFARDataArgs, CIFAR100DataModule


@dataclass
class CIFARLowDataDataArgs(CIFARDataArgs):
    """Data arguments for CIFAR low-data environment
    Args:
        train_size (float): size of the int set of each class (from train)
        k_way (int): number of classes to classify on
    """

    train_size: int = 1600 # By default, train set will be compliment of val set. 
    k_way: int = 100

    # filled in for you in __post_init__()
    metadata: dict = None # CIFAR100 metadata
    classes: tuple = None  # all classes built from metadata

    def __post_init__(self):
        super().__post_init__()
        self.train_classes = tuple(self.classes[0:self.k_way])
        self.val_classes = self.train_classes


class CIFARLowDataDM(CIFAR100DataModule):
    """class which iplements few-shot pretraining on cifar"""

    def __init__(self, args: CIFARLowDataDataArgs, *margs, **kwargs):
        super().__init__(args, *margs, **kwargs)
    
    def setup(self, stage=None):
        self.setup_labels()

        # get the dataset
        train_dataset = torchvision.datasets.CIFAR100(
            root=self.args.cache_dir, train=True, download=False
        )
        val_dataset = torchvision.datasets.CIFAR100(
            root=self.args.cache_dir, train=True, download=False
        )

       


         # make stratified split of train and val datasets
        train_idx, val_idx = train_test_split(
            list(range(len(train_dataset))),
            train_size=self.args.train_size,
            test_size=self.args.val_size,
            random_state=self.args.split_seed,
            shuffle=True,
            stratify=train_dataset.targets,
        )
        #train_dataset = Subset(train_dataset, train_idx)
        #val_dataset = Subset(val_dataset, val_idx)


        
        # class_ids that should not be in training / val
        train_heldout_ids = [
            self.all_classlabel.str2int(x) for x in self.args.classes if x not in self.args.train_classes
        ]
        val_heldout_ids = train_heldout_ids

        # filter the train and val examples
        train_idx_heldout, val_idx_heldout = [], []
        for idx in range(len(train_dataset)):
            _, class_id = train_dataset[idx]
            if class_id in train_heldout_ids:
                continue
            train_idx_heldout.append(idx)

        for idx in range(len(val_dataset)):
            _, class_id = val_dataset[idx]
            if class_id in val_heldout_ids:
                continue
            val_idx_heldout.append(idx)


        def intersection(L1, L2):
            return list(set(L1).intersection(L2))

        train_idx_keep = intersection(train_idx_heldout, train_idx)
        val_idx_keep = intersection(val_idx_heldout, val_idx)

        train_dataset = Subset(train_dataset, train_idx_keep)
        val_dataset = Subset(val_dataset, val_idx_keep)

       
       # get the label transforms for FSL
        def train_target_transform(class_id: int) -> int:
            # converts the id from the original class label to the correct zsl training set label
            return self.train_classlabel.str2int(self.all_classlabel.int2str(class_id))

        def val_target_transform(class_id: int) -> int:
            return self.val_classlabel.str2int(self.all_classlabel.int2str(class_id))

        # add transforms to the datasets
        # we can't do this earlier, because we need the original target_transforms to do the
        # heldout sets
        val_dataset.dataset.transform = self.eval_transform
        val_dataset.dataset.target_transform = val_target_transform
        val_dataset.dataset.transforms = StandardTransform(
            transform=self.eval_transform, target_transform=val_target_transform
        )

        train_dataset.dataset.transform = self.train_transform
        train_dataset.dataset.target_transform = train_target_transform
        train_dataset.dataset.transforms = StandardTransform(
            transform=self.train_transform, target_transform=train_target_transform
        )

        self.dataset["train"] = train_dataset
        self.dataset["val"] = val_dataset