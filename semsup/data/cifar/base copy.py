""" base CIFAR100 datamodule
"""
import pickle
from dataclasses import dataclass
from pathlib import Path
import numpy as np #new import

from datasets import ClassLabel
from torch.utils.data import Subset
from torchvision import transforms
import torchvision
from sklearn.model_selection import train_test_split

from ..core import SemSupDataArgs, SemSupDataModule


@dataclass
class CIFARDataArgs(SemSupDataArgs):
    """Base argument dataclass for all CIFAR100 tasks
    Args:
        split_seed (int): seed to make train-val split
        test_size (float): size of the test size of each class (from total)
        val_size (float): size of the val set of each class (from train)
        train_size (float): size of the int set of each class (from train)
        load_test (bool): load the test dataset
    """

    split_seed: int = 42  # seed to make train-val split
    train_size = None # By default, train set will be compliment of val set. 
    val_size: float = 0.2  # size of the val set of each class (from train)
    load_test: bool = False  # load the test datset

    # filled in for you in __post_init__()
    metadata: dict = None # CIFAR100 metadata
    classes: tuple = None  # all classes built from metadata

    def __post_init__(self):
        super().__post_init__()

        metadata_path = Path(self.cache_dir).joinpath("cifar-100-python", "meta")
        if not metadata_path.exists():
            torchvision.datasets.CIFAR100(
                root=self.cache_dir, train=True, download=True
            )

        with metadata_path.open(mode="rb") as f:
            self.metadata = pickle.load(f, encoding="bytes")
        self.classes = tuple(
            [c.decode("utf-8") for c in self.metadata[b"fine_label_names"]]
        )
        self.train_classes = self.classes
        self.val_classes = self.train_classes


class CIFAR100DataModule(SemSupDataModule):
    """Pytorch lightning datamodule"""

    def __init__(
        self,
        args: CIFARDataArgs,
        batch_size: int = 64,
        num_workers: int = 0,
        val_batch_size: int = None,
        **kwargs
    ):
        super().__init__(
            args=args,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            **kwargs,
        )
        # classlabel for all classes
        self.all_classlabel = ClassLabel(names=self.args.classes)

        # CIFAR stats
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]

        # transforms
        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        self.eval_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]
        )

    def prepare_data(self):
        self.prepare_label_data()
        torchvision.datasets.CIFAR100(
            root=self.args.cache_dir, train=True, download=True
        )
        torchvision.datasets.CIFAR100(
            root=self.args.cache_dir, train=False, download=True
        )

    def setup(self, stage=None):
        self.setup_labels()
        train_dataset = torchvision.datasets.CIFAR100(
            root=self.args.cache_dir,
            train=True,
            download=False,
            transform=self.train_transform,
        )
        val_dataset = torchvision.datasets.CIFAR100(
            root=self.args.cache_dir,
            train=True,
            download=False,
            transform=self.eval_transform,
        )

         #new masking code:

        #remove data points that don't belong to the classes specified by self.args.classes
        #train_class_mask should be same as val_class_mask
        train_class_mask = np.array([1 if train_dataset[idx][1] in [self.all_classlabel.str2int(class_str) for class_str in self.args.classes] else 0 for idx in range(len(train_dataset))], dtype = bool)
        val_class_mask = np.array([1 if val_dataset[idx][1] in [self.all_classlabel.str2int(class_str) for class_str in self.args.classes] else 0 for idx in range(len(val_dataset))], dtype = bool)
        print("before: ",len(train_dataset.data))
        print("n_true: ", np.sum(train_class_mask))
        train_dataset.data = train_dataset.data[train_class_mask]
        print("after: ", len(train_dataset.data))
        train_dataset.targets = np.array(train_dataset.targets)[train_class_mask]
        val_dataset.data = val_dataset.data[val_class_mask]
        val_dataset.targets = np.array(val_dataset.targets)[val_class_mask]

        if self.args.run_test:
            self.test_dataset = torchvision.datasets.CIFAR100(
                root=self.args.cache_dir,
                train=False,
                download=False,
                transform=self.eval_transform,
            )
            test_class_mask = np.array([1 if self.test_dataset[idx][1] in [self.all_classlabel.str2int(class_str) for class_str in self.args.classes] else 0 for idx in range(len(self.test_dataset))], dtype = bool)
            self.test_dataset.data = np.array(self.test_dataset.data)[test_class_mask]
            self.test_dataset.targets = self.test_dataset.targets[test_class_mask]
            self.dataset["test"] = self.test_dataset

        # make stratified split of train and val datasets
        train_idx, val_idx = train_test_split(
            list(range(len(train_dataset))),
            train_size=self.args.train_size,
            test_size=self.args.val_size,
            random_state=self.args.split_seed,
            shuffle=True,
            stratify=train_dataset.targets,
        )
        self.dataset["train"] = Subset(train_dataset, train_idx)
        self.dataset["val"] = Subset(val_dataset, val_idx)

        if self.args.run_test:
            self.dataset["val"] = self.dataset["test"]
        
        print("setup complete")
        print("train len: ", len(self.dataset["train"]))
        print("val len: ", len(self.dataset["val"]))
        if self.args.run_test:
            print("test len: ", len(self.dataset["test"]))


if __name__ == "__main__":
    # unit tests
    data_args = CIFARDataArgs(
        label_tokenizer="prajjwal1/bert-small",
        train_label_json="../class_descs/cifarGPT_formatted.labels",
        cache_dir="../data_cache",
    )
    data_mod = CIFAR100DataModule(args=data_args)
    data_mod.prepare_data()
    data_mod.setup()
    for _ in data_mod.train_dataloader():
        pass
    for _ in data_mod.val_dataloader():
        pass
