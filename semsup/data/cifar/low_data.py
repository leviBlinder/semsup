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
    """

    train_size: int = 16 # By default, train set will be compliment of val set. 

    # filled in for you in __post_init__()
    metadata: dict = None # CIFAR100 metadata
    classes: tuple = None  # all classes built from metadata

    def __post_init__(self):
        super().__post_init__()


class CIFARLowDataDM(CIFAR100DataModule):
    """class which iplements few-shot pretraining on cifar"""

    def __init__(self, args: CIFARLowDataDataArgs, *margs, **kwargs):
        super().__init__(args, *margs, **kwargs)
