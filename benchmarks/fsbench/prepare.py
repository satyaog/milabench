#!/usr/bin/env python
import os
import subprocess

from milabench.datasets.fake_images import generate_sets
from milabench.fs import XPath
import numpy as np

try:
    import deeplake

    can_prepare_deeplake = True
except ModuleNotFoundError:
    can_prepare_deeplake = False

try:
    import h5py
    import jug
    import jug.mapreduce
    import torchvision
    from jug import TaskGenerator


    def CachedFunction(f, *args, **kwargs):
        from jug import CachedFunction as _CachedFunction
        if isinstance(f, TaskGenerator):
            return _CachedFunction(f.f, *args, **kwargs)
        else:
            return _CachedFunction(f, *args, **kwargs)
    

    def read_image(path: str) -> bytes:
        with open(path, 'rb') as _f:
            return _f.read()


    @TaskGenerator
    def add_image(i: int, split: str, dataset: torchvision.datasets.ImageFolder, dest: str):
        dataset.loader = read_image
        with h5py.File(dest, 'a') as h5f:
            b, l = dataset[i]
            h5f[f"/{split}_images"][i] = np.frombuffer(b, dtype='uint8')
            h5f[f"/{split}_labels"][i] = l

    
    can_prepare_h5 = True
except ModuleNotFoundError:
    can_prepare_h5 = False

try:
    import bcachefs
    import benzina.torch as B
    subprocess.run(["git-annex", "version"], check=True)

    can_prepare_bch = True
except ModuleNotFoundError:
    can_prepare_bch = False



def make_deeplake_group(ds, folder, class_names):
    group_name = os.path.basename(folder)

    files_list = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(folder)):
        for filename in filenames:
            if filename == 'done':
                continue
            files_list.append(os.path.join(dirpath, filename))

    with ds:
        ds.create_group(group_name)
        ds[group_name].create_tensor('images', htype='image', sample_compression='jpeg')
        ds[group_name].create_tensor('labels', htype='class_label', class_names=class_names)
        for f in files_list:
            label_num = int(os.path.basename(os.path.dirname(f)))
            ds[group_name].append({'images': deeplake.read(f), 'labels': np.uint16(label_num)})


def generate_h5(src: str, dest: str):
    jug.init(__file__, dest + ".jugdir")

    datasets = {}
    for split in ("test", "train", "val"):
        datasets[split] = CachedFunction(torchvision.datasets.ImageFolder, os.path.join(src, f"{split}/"), loader=None)

    with h5py.File(dest, 'a') as h5f:
        for split in ("test", "train", "val"):
            try:
                _ = h5f[f"/{split}_images"]
                _ = h5f[f"/{split}_labels"]
            except KeyError:
                _ = h5f.create_dataset(f"{split}_images", shape=(len(datasets[split]),),
                                       dtype=h5py.special_dtype(vlen=np.uint8))
                _ = h5f.create_dataset(f"{split}_labels", shape=(len(datasets[split]),),
                                       dtype="i4")

    tasklets: list[jug.Tasklet] = jug.mapreduce.currymap(add_image,
        [(i, s, datasets[s], dest)
         for s in datasets for i in range(len(datasets[s]))],
        map_step=1024)
    
    for t in tasklets:
        task: jug.Task = t.base
        if not task.can_load() and task.can_run() and task.lock():
            try:
                task.run()
            finally:
                task.unlock()

    return all(t.can_load() for t in tasklets)


def generate_bch(src: str, dest: str):
    bch_root = os.path.join(os.environ["MILABENCH_DIR_DATA"], "bcachefs")
    if not os.path.isdir(bch_root):
        subprocess.run(["git", "clone", "https://github.com/mila-iqia/bcachefs.git", bch_root], check=True)
        subprocess.run(["git-annex", "get", "scripts/"], cwd=bch_root, check=True)
    bch_make_disk_image = os.path.join(bch_root, "scripts", "make_disk_image.sh")
    size = "-1" if os.path.isfile(dest) else ""
    rm_failed = "1" if os.path.isfile(dest) else ""
    p = subprocess.Popen([bch_make_disk_image], stdout=subprocess.PIPE,
            env={**os.environ, "NAME": dest, "CONTENT_SRC": src, "SIZE": size, "RM_FAILED": rm_failed, "TMP_DIR": f"{dest}_tmp"})
    failed = False
    line = p.stdout.readline()
    while line:
        if b"Run the following commands in another shell:" in line:
            # Next line is the command
            line = p.stdout.readline()
            cwd = line.split(b" && ")[0].split(b"pushd ")[-1].strip(b"'\"")
            try:
                subprocess.run(["./cp.sh", "."], cwd=cwd, check=True,
                        env={**os.environ, "UNMOUNT": "1", "RM_FAILED": rm_failed})
            except subprocess.CalledProcessError:
                subprocess.run(["./unmount.sh"], cwd=cwd, check=True)
                failed = True
            break
        line = p.stdout.readline()
    return not failed and p.wait() == 0


# adjust the size of the generated dataset (1 = ~500Mb)
scale = 400
if __name__ == "__main__":
    # If you need the whole configuration:
    # config = json.loads(os.environ["MILABENCH_CONFIG"])

    data_directory = os.environ["MILABENCH_DIR_DATA"]
    dest = os.path.join(data_directory, "LargeFakeUniform")

    generate_sets(dest, {"train": 14000 * scale, "val": 500 * scale, "test": 500 * scale}, (3, 256, 256))

    sentinel = XPath(dest + ".img" + '-done')
    if sentinel.exists():
        print(f"{dest}.img was already generated")
    elif can_prepare_bch:
        if generate_bch(dest, dest + ".img"):
            sentinel.touch()

    sentinel = XPath(dest + ".h5" + '-done')
    if sentinel.exists():
        print(f"{dest}.h5 was already generated")
    elif can_prepare_h5:
        if generate_h5(dest, dest + ".h5"):
            sentinel.touch()

    root = dest + '.lake'
    sentinel = XPath(root + '-done')
    if sentinel.exists():
        print(f"{root} was already generated")
    elif can_prepare_deeplake:
        ds = deeplake.empty(dest + '.lake')
        class_names = [str(i) for i in range(1000)]
        make_deeplake_group(ds, os.path.join(dest, 'train'), class_names)
        make_deeplake_group(ds, os.path.join(dest, 'val'), class_names)
        make_deeplake_group(ds, os.path.join(dest, 'test'), class_names)
        sentinel.touch()
