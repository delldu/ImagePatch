"""Data loader."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 02日 星期一 17:49:55 CST
# ***
# ************************************************************************************/
#

import os
import pdb

import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.utils as utils
from PIL import Image

train_dataset_rootdir = "dataset/train/"
test_dataset_rootdir = "dataset/test/"


def get_transform(train=True):
    """Transform images."""
    ts = []
    if train:
        loadSize = 256
        ts.append(T.Resize(size=loadSize, interpolation=Image.BICUBIC))
    #   ts.append(T.RandomHorizontalFlip(0.5))

    ts.append(T.ToTensor())
    return T.Compose(ts)


def image_with_mask(image, mask):
    """image, mask: NX3xHxW."""
    threshhold = 0.5
    ones = mask >= threshhold
    zeros = mask < threshhold
    mask.masked_fill_(ones, 1.0)
    mask.masked_fill_(zeros, 0.0)
    mask = 1 - mask
    image = image * mask
    image = torch.cat((image, mask[:, 0:1, :, :]), 1)
    # pdb.set_trace()
    # (Pdb) image.size()
    # torch.Size([1, 4, 1024, 1024])
    # (Pdb) mask.size()
    # torch.Size([1, 3, 1024, 1024])
    
    return image, mask


class ImagePatchDataset(data.Dataset):
    """Define dataset."""

    def __init__(self, root, transforms=get_transform()):
        """Init dataset."""
        super(ImagePatchDataset, self).__init__()

        self.root = root
        self.transforms = transforms

        # load all images, sorting for alignment
        self.images = list(sorted(os.listdir(root + "/image")))

    def __getitem__(self, idx):
        """Load images."""
        image_path = os.path.join(self.root + "/image", self.images[idx])
        image = Image.open(image_path).convert("RGB")
        mask_path = os.path.join(self.root + "/mask", self.images[idx])
        mask = Image.open(mask_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return image, mask

    def __len__(self):
        """Return total numbers of images."""
        return len(self.images)

    def __repr__(self):
        """
        Return printable representation of the dataset object.
        """
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms: '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def train_data(bs):
    """Get data loader for trainning & validating, bs means batch_size."""

    train_ds = ImagePatchDataset(
        train_dataset_rootdir, get_transform(train=True))
    print(train_ds)

    # Split train_ds in train and valid set
    valid_len = int(0.2 * len(train_ds))
    indices = [i for i in range(len(train_ds) - valid_len, len(train_ds))]

    valid_ds = data.Subset(train_ds, indices)
    indices = [i for i in range(len(train_ds) - valid_len)]
    train_ds = data.Subset(train_ds, indices)

    # Define training and validation data loaders
    train_dl = data.DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=4)
    valid_dl = data.DataLoader(
        valid_ds, batch_size=bs, shuffle=False, num_workers=4)

    return train_dl, valid_dl


def test_data(bs):
    """Get data loader for test, bs means batch_size."""

    test_ds = ImagePatchDataset(
        test_dataset_rootdir, get_transform(train=False))
    test_dl = data.DataLoader(test_ds, batch_size=bs,
                              shuffle=False, num_workers=4)

    return test_dl


def get_data(trainning=True, bs=4):
    """Get data loader for trainning & validating, bs means batch_size."""

    return train_data(bs) if trainning else test_data(bs)


def ImagePatchDatasetTest():
    """Test dataset ..."""

    ds = ImagePatchDataset(train_dataset_rootdir)
    print(ds)
    # src, tgt = ds[10]
    # grid = utils.make_grid(torch.cat([src.unsqueeze(0), tgt.unsqueeze(0)], dim=0), nrow=2)
    # ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # image = Image.fromarray(ndarr)
    # image.show()


if __name__ == '__main__':
    ImagePatchDatasetTest()
