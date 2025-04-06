import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # If the folder exists, delete the original folder and then recreate it.
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    random.seed(0)

    # How much data will be divided into the test set
    split_rate = 0.1

    cwd = os.getcwd()
    data_root = os.path.join(cwd, "ImageNet")
    origin_data_path = os.path.join(data_root, "val")
    assert os.path.exists(origin_data_path), "path '{}' does not exist.".format(origin_data_path)

    flower_class = [cla for cla in os.listdir(origin_data_path)
                    if os.path.isdir(os.path.join(origin_data_path, cla))]

    # Val set
    train_root = os.path.join(data_root, "test0p9")
    mk_file(train_root)
    for cla in flower_class:
        mk_file(os.path.join(train_root, cla))

    # Test set
    val_root = os.path.join(data_root, "test0p1")
    mk_file(val_root)
    for cla in flower_class:
        mk_file(os.path.join(val_root, cla))

    for cla in flower_class:
        cla_path = os.path.join(origin_data_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        eval_index = random.sample(images, k=int(num*split_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main()
