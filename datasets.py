
"""
# modified from https://github.com/IBM/CrossViT
"""

import os
import json
import glob

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


def default_loader(path):
    img = Image.open(path).convert('RGB')
    return img

        
class GAFDataset(Dataset):
    def __init__(self, is_val, img_root, cropped_img_root, txt_list, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        for txt in txt_list:
            fh = open(txt, 'r')
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split(' ')
                target = -1
                #print(words[0])
                Emotion = words[0].split('/')[0]
                if Emotion == "Negative":
                    target = 0
                elif Emotion == "Neutral":
                    target = 1
                elif Emotion == "Positive":
                    target = 2
                img_num = "/Image_" + words[0].split('.')[0].split('_')[-1].zfill(4)
                if words[1] == "-1":
                    MIP_img = img_root+words[0]
                else:
                    MIP_img = cropped_img_root + Emotion + img_num + "/Face" + img_num + "_Face_" + words[1].zfill(2) + "_Label_0.jpg"
                
                imgs.append((img_root+words[0], MIP_img, target))     # (fullImg, mipImg, gt)
                
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.Len = len(imgs)

    def __getitem__(self, index):
        fn1, fn2, label = self.imgs[index]
        fullImg = self.loader(fn1)
        mipImg = self.loader(fn2)
        if self.transform is not None:
            fullImg = self.transform(fullImg)
            mipImg = self.transform(mipImg)
        return fullImg, mipImg, label

    def __len__(self):
        return self.Len


class GroupEmoWDataset(Dataset):
    def __init__(self, is_val, img_root, cropped_img_root, txt_list, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        for txt in txt_list:
            fh = open(txt, 'r')
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split(' ')
                target = -1
                #print(words[0])
                Emotion = words[0].split('/')[0]
                if Emotion == "Negative":
                    target = 0
                elif Emotion == "Neutral":
                    target = 1
                elif Emotion == "Positive":
                    target = 2
                img_num = "/Image_" + words[0].split('.')[0].split('_')[-1].zfill(4)
                if words[1] == "-1":
                    MIP_img = img_root+words[0]
                else:
                    MIP_img = cropped_img_root + Emotion + img_num + "/Face" + img_num + "_Face_" + words[1].zfill(2) + "_Label_0.jpg"
                
                img_name1 = img_root+words[0].split('.')[0]
                img_name2 = glob.glob(img_name1+"*")
                imgs.append((img_name2[0], MIP_img, target))     # (fullImg, mipImg, gt)
                
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.Len = len(imgs)

    def __getitem__(self, index):
        fn1, fn2, label = self.imgs[index]
        fullImg = self.loader(fn1)
        mipImg = self.loader(fn2)
        if self.transform is not None:
            fullImg = self.transform(fullImg)
            mipImg = self.transform(mipImg)
        return fullImg, mipImg, label

    def __len__(self):
        return self.Len


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform)
        nb_classes = 10
    elif args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    ###    
    elif args.data_set == 'GAF':
        if is_train:
            root = args.mip_root_train
            cropped_root = args.mip_cropped_root_train
            txt_root = args.mip_txt_root_train
            txtl = []    
            for Dir in glob.glob(txt_root+"*"):
                txtl.append(Dir+"/result.txt")
        
        else:
            root = args.mip_root_val
            cropped_root = args.mip_cropped_root_val
            txt_root = args.mip_txt_root_val
            txtl = []    
            for Dir in glob.glob(txt_root+"*"):
                txtl.append(Dir+"/result.txt")
            
        dataset = GAFDataset(is_val=False, img_root=root, cropped_img_root=cropped_root, txt_list=txtl, transform=transform)
        nb_classes = 3
        
    elif args.data_set == 'GroupEmoW':
        if is_train:
            root = args.mip_root_train
            cropped_root = args.mip_cropped_root_train
            txt_root = args.mip_txt_root_train
            txtl = []    
            for Dir in glob.glob(txt_root+"*"):
                txtl.append(Dir+"/result.txt")
        
        else:
            root = args.mip_root_val
            cropped_root = args.mip_cropped_root_val
            txt_root = args.mip_txt_root_val
            txtl = []    
            for Dir in glob.glob(txt_root+"*"):
                txtl.append(Dir+"/result.txt")
            
        dataset = GroupEmoWDataset(is_val=False, img_root=root, cropped_img_root=cropped_root, txt_list=txtl, transform=transform)
        nb_classes = 3
    ###

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
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
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.crop_ratio * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
