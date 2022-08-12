from .newsgroups.base import NewsgroupsDataModule, NewsgroupsDataArgs
from .newsgroups.superclass import NewsgroupsSuperClassDM, NewsgroupsSuperClassArgs
from .newsgroups.heldout import NewsgroupsHeldoutArgs, NewsgroupsHeldoutDM

from .cifar.base import CIFAR100DataModule, CIFARDataArgs
from .cifar.heldout import CIFARHeldoutDM, CIFARHeldoutDataArgs
from .cifar.low_data import CIFARLowDataDM, CIFARLowDataDataArgs
from .cifar.fewshot import CIFARFewShotDM, CIFARFewShotDataArgs
from .cifar.fewshot_pretrain import CIFARFewShotPretrainDM, CIFARFewShotPretrainDataArgs
from .cifar.superclass import CIFARSuperClassDataArgs, CIFARSuperClassDM

from .awa.base import AWADataModule, AWADataArgs
from .awa.heldout import AWAHeldoutArgs, AWAHeldoutDM