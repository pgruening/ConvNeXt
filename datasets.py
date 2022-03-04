# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import os

from timm.data import create_transform
from timm.data.constants import (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD,
                                 IMAGENET_INCEPTION_MEAN,
                                 IMAGENET_INCEPTION_STD)
from torchvision import datasets, transforms
from torch.utils.data import Subset


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(
            args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    elif args.data_set == "CALTECH":
        dataset = datasets.ImageFolder(
            'data/caltech101/101_ObjectCategories',
            transform=transform
        )

        nb_classes = 101

        # load train or test set in accordance with the indices we've defined
        with open(os.path.join('data', 'caltech_split.json'), 'r') as file:
            indices = json.load(file)

        is_train = 'train' if is_train else 'test'
        dataset = Subset(dataset, indices[is_train])

    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:
            t.append(
                transforms.Resize((args.input_size, args.input_size),
                                  interpolation=transforms.InterpolationMode.BICUBIC),
            )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BICUBIC),
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def create_caltech_train_test_split():
    import numpy as np

    dataset = datasets.ImageFolder('data/caltech101/101_ObjectCategories')
    labels = dataset.targets
    class_range = {}
    added_classes = []
    first_entry = True
    for index, y in enumerate(labels):
        if y not in class_range.keys():
            class_range[y] = [index]
            added_classes.append(y)
            if not first_entry:
                class_range[added_classes[-2]].append(index - 1)
                assert added_classes[-1] - added_classes[-2] == 1
            else:
                first_entry = False

    # last entry
    class_range[y].append(len(labels) - 1)

    # percentage of test samples per class
    p_test = .2
    test_indices = []
    train_indices = []
    for range_ in class_range.values():
        tmp = np.arange(range_[0], range_[1] + 1)
        perm = np.random.permutation(tmp.shape[0])
        n = int(p_test * tmp.shape[0])
        test_indices += list(tmp[perm][:n])
        train_indices += list(tmp[perm][n:])

    assert np.abs(len(test_indices) - p_test * len(labels)) < 50
    assert not set(train_indices).intersection(set(test_indices))

    with open(os.path.join('data', 'caltech_split.json'), 'w') as file:
        json.dump({
            'train': [int(x) for x in train_indices],
            'test': [int(x) for x in test_indices]
        }, file)


if __name__ == '__main__':
    create_caltech_train_test_split()
