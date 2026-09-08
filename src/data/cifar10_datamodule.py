import os
from typing import Any, Dict, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import transforms


class CIFAR10DataModule(LightningDataModule):
    """`LightningDataModule` for the CIFAR-10 dataset.

    CIFAR-10 consists of 60,000 32x32 colour images in 10 classes, with 6,000 images per
    class. There are 50,000 training images and 10,000 test images.

    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 128,
        num_workers: int = 8,
        pin_memory: bool = True,
        val_from_train: bool = False,
    ) -> None:
        """Initialize a `CIFAR10DataModule`.

        :param data_dir: The base data directory. Contains the downloaded archive under
            `raw/` and the extracted dataset under `processed/`. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `128`.
        :param num_workers: The number of workers. Defaults to `8`.
        :param pin_memory: Whether to pin memory. Defaults to `True`.
        :param val_from_train: Whether to carve validation out of the train set instead of
            reusing the test set. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        self.test_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of CIFAR-10 classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        raw_dir = os.path.join(self.hparams.data_dir, "raw")
        processed_dir = os.path.join(self.hparams.data_dir, "processed")
        # tgz -> data/raw/, extracted cifar-10-batches-py -> data/processed/
        if not os.path.isdir(os.path.join(processed_dir, CIFAR10.base_folder)):
            download_and_extract_archive(
                url=CIFAR10.url,
                download_root=raw_dir,
                extract_root=processed_dir,
                filename=CIFAR10.filename,
                md5=CIFAR10.tgz_md5,
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            processed_dir = os.path.join(self.hparams.data_dir, "processed")
            trainset = CIFAR10(
                processed_dir, train=True, transform=self.train_transforms, download=False
            )
            testset = CIFAR10(
                processed_dir, train=False, transform=self.test_transforms, download=False
            )
            if self.hparams.val_from_train:
                # NOTE: this 5k validation split still carries train-time augmentation.
                self.data_train, self.data_val = random_split(
                    dataset=trainset,
                    lengths=(45_000, 5_000),
                    generator=torch.Generator().manual_seed(42),
                )
                self.data_test = testset
            else:
                # Reuse the test set for validation-curve monitoring: reported val/test metrics coincide.
                self.data_train = trainset
                self.data_val = testset
                self.data_test = testset

    def _loader_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "batch_size": self.batch_size_per_device,
            "num_workers": self.hparams.num_workers,
            "pin_memory": self.hparams.pin_memory,
        }
        if self.hparams.num_workers > 0:
            # keep workers alive across epochs (avoids per-epoch respawn stalls) and prefetch more
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4
        return kwargs

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(dataset=self.data_train, shuffle=True, **self._loader_kwargs())

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(dataset=self.data_val, shuffle=False, **self._loader_kwargs())

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(dataset=self.data_test, shuffle=False, **self._loader_kwargs())

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = CIFAR10DataModule()
