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
        few_way_k (int): Sets number of heldout classes to train/validate on. 
    """
    fewshot_names: tuple = (
        "motorcycle",
        "pine_tree",
        "bottle",
        "trout",
        "chair"
    )
    # fewshot_names: tuple = (
    #     "streetcar",
    #     "lamp",
    #     "forest",
    #     "otter",
    #     "house",
    #     "crab",
    #     "crocodile",
    #     "orchid",
    #     "rabbit",
    #     "man",
    # )
    few_way_k: int = 0
    few_shot_k: int = 0 # Number of samples from heldout classes to include in training, should be overidden



    def __post_init__(self):
        super().__post_init__()
        if self.few_way_k == 10:
            self.fewshot_names = (
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
        self.train_size = int(self.few_shot_k * len(self.fewshot_names))
        self.train_size = int(self.few_shot_k * 100)
        self.val_size = None
        print("fewshot_k: ", self.few_shot_k)
        print("few_way_k: ", self.few_way_k)
        print("train_size: ", self.train_size)


        self.train_classes = tuple(
            [x for x in self.classes if x in self.fewshot_names]
        )

        self.val_classes = self.train_classes
        self.val_names = self.train_classes


class CIFARFewShotDM(CIFAR100DataModule):
    """class which iplements zero-shot cifar"""

    def __init__(self, args: CIFARFewShotDataArgs, *margs, **kwargs):
        super().__init__(args, *margs, **kwargs)
        #self.batch_size = 5

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
        val_heldout_ids = [
            self.all_classlabel.str2int(x) for x in self.args.classes if x not in self.args.val_classes
        ]

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
        print("PRE-TRANSFORM")
        train_dataset.dataset.transform = self.train_transform
        train_dataset.dataset.target_transform = train_target_transform
        train_dataset.dataset.transforms = StandardTransform(
            transform=self.train_transform, target_transform=train_target_transform
        )

        print("POST-TRANSFORM")
        # self.dataset["train"] = train_dataset.dataset
        # self.dataset["val"] = val_dataset.dataset
        # print("train transform: ", self.dataset["train"].transform)
        # print("val transform: ", self.dataset["val"].transform)
        self.dataset["train"] = train_dataset
        self.dataset["val"] = val_dataset
        print("train transform: ", self.dataset["train"].dataset.transform)
        print("val transform: ", self.dataset["val"].dataset.transform)
        print("train data: ", train_dataset)
        
        #sanity checks
        print("train classes: ", self.args.train_classes)
        print("val classes: ", self.args.val_classes)
        #print("classes: ", self.args.classes)
        print("train length: ", len(self.dataset["train"]))
        print("val length: ", len(self.dataset["val"]))
        #print("test length: ", len(self.dataset["train"]))
        #class_ids = [self.all_classlabel.str2int(class_name) for class_name in self.args.classes]
        # assert self.args.train_classes == self.args.val_classes == self.args.fewshot_names, "Mismatched classes"
        # for idx in range(len(self.dataset["train"])):
        #     _, class_id = self.dataset["train"][idx]
        #     assert class_id in class_ids, f"In train dataset, {class_id} not a valid class id"
        # for idx in range(len(self.dataset["val"])):
        #     _, class_id = self.dataset["val"][idx]
        #     assert class_id in class_ids, f"In val dataset, {class_id} not a valid class id"

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
