import argparse
import io
import os

from giving import give
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision.transforms import ToTensor

try:
    import h5py
except ModuleNotFoundError:
    pass

try:
    import bcachefs as bch
    import benzina.torch as B
except ModuleNotFoundError:
    pass


class SqhDataset(Dataset):
    def __init__(self, root):
        self.root = root
        labels = list(root)
        self.files = []
        for label in labels:
            self.files.extend(map(lambda p: label + b"/" + p, root.cd(label)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = int(path.split(b"/")[0])
        return len(self.root.open(path, binary=True, buffering=0).readall()), label


class H5Dataset(Dataset):
    def __init__(self, filename: str, split: str):
        self._file: "h5py.File" = None
        self.filename = filename
        self.split = split
        with h5py.File(filename, "r") as h5f:
            self._len = len(h5f[f"/{split}_images"])

    def __len__(self):
        with h5py.File(self.filename, "r") as h5f:
            return len(h5f[f"/{self.split}_images"])

    def __getitem__(self, idx):
        return len(io.BytesIO(self.ds[idx][:]).read()), int(self.labels[idx])
    
    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(self.filename, "r")
        return self._file
    
    @property
    def ds(self):
        return self.file[f"/{self.split}_images"]

    @property
    def labels(self):
        return self.file[f"/{self.split}_labels"]


def _ld(path):
    with open(path, "rb") as f:
        return len(f.read())


def make_loader(path, sub, shuffle, batch_size, loading_processes, load_type):
    if load_type == "fs":
        load = DatasetFolder(os.path.join(path, sub), loader=_ld, extensions=(".jpeg",))
    elif load_type == "squash":
        from pysquash import SquashCursor

        load = SqhDataset(SquashCursor(path + ".sqh").cd(sub.encode("utf-8")))
    elif load_type == "bcache":
        with bch.Bcachefs(path + ".img") as bchfs:
            load = B.dataset.ClassificationDatasetMixin(bchfs.cd(sub))
    elif load_type == "hdf5":
        load = H5Dataset(path + ".h5", sub)
    elif load_type == "deeplake":
        import deeplake
        return deeplake.load(path + ".lake")[sub].pytorch(batch_size=batch_size, shuffle=shuffle, num_workers=loading_processes, tensors=["images", "labels"], return_index=False, tobytes=True)
    elif load_type == "decode":
        load = ImageFolder(path, transform=ToTensor())
    else:
        raise ValueError("unknown load_type")
    return DataLoader(
        load, batch_size=batch_size, shuffle=shuffle, num_workers=loading_processes
    )


def main():
    parser = argparse.ArgumentParser(description="Filesystem benchmarks")
    parser.add_argument(
        "--shuffle", type=bool, default=False, help="read dataset in a random order"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--epochs-valid",
        type=int,
        default=10,
        help="number of epochs between validations",
    )
    parser.add_argument(
        "--iters", type=int, default=1, help="number of train/valid cycles to run"
    )
    parser.add_argument(
        "--loading-processes",
        type=int,
        default=0,
        help="number of external processes to use for loading (0 to disable)",
    )
    parser.add_argument(
        "--load-type",
        required=True,
        choices=("fs", "squash", "bcache", "hdf5", "deeplake", "decode"),
        help="type of loading to test, 'fs' is raw filesystem with no decode, 'squash' is loading trough squashfile, and 'decode' if from the filesystem, but including image decode",
    )

    args = parser.parse_args()

    data_directory = os.environ.get("MILABENCH_DIR_DATA", None)
    dataset_dir = os.path.join(data_directory, "LargeFakeUniform")

    train_loader = make_loader(
        dataset_dir,
        "train",
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        loading_processes=args.loading_processes,
        load_type=args.load_type,
    )
    valid_loader = make_loader(
        dataset_dir,
        "val",
        shuffle=False,
        batch_size=args.batch_size,
        loading_processes=args.loading_processes,
        load_type=args.load_type,
    )
    test_loader = make_loader(
        dataset_dir,
        "test",
        shuffle=False,
        batch_size=args.batch_size,
        loading_processes=args.loading_processes,
        load_type=args.load_type,
    )

    for _ in range(args.iters):
        for epoch in range(args.epochs_valid):
            for inp, target in train_loader:
                give(batch=inp, step=True)
        for inp, target in valid_loader:
            give(batch=inp, step=True)
    for inp, target in test_loader:
        give(batch=inp, step=True)

if __name__ == "__main__":
    # Note: The line `if __name__ == "__main__"` is necessary for milabench
    # to recognize the entry point (it does some funky stuff to it).
    main()
