"""
Training Script for:
    "Random Shadows and Highlights: A new data augmentation method for extreme lighting conditions"
    "https://arxiv.org/abs/2101.05361"
Copyright (c) 2021 Osama MAZHAR and Jens KOBER
Licensed under the Apache License 2.0 (See LICENSE for details)
Originally Written by Osama Mazhar.
"""
import numpy as np
import torch, os
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torchvision.transforms as transforms
import argparse

from train_model import train_model
from AlexNet import alexnet
# train model, AlexNet and some parts of this code are taken from:
# https://github.com/tjmoon0104/pytorch-tiny-imagenet
# and are modified wherever necessary.

from efficientnet_pytorch import EfficientNet
# https://github.com/lukemelas/EfficientNet-PyTorch

from DiskAugmenter import Augmenter as DiskAugmenter
from RandomShadowsHighlights import RandomShadows
from OtherTransforms import RandomGamma, RandomColorJitter

if __name__=="__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=100)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=100)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=0.001)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default="./saved_models")
    parser.add_argument("--dataset", type=str, help="TinyImageNet, CIFAR-10 or CIFAR100", default="TinyImageNet")
    parser.add_argument("--data_dir", type=str, help="TinyImageNet dataset directory", default="../pytorch-cifar/tiny-imagenet-200/")
    parser.add_argument("--model_name", type=str, help="EfficientNet, AlexNet", default="EfficientNet")
    parser.add_argument("--filename", type=str, help="filename for logging.", default="output.txt")
    parser.add_argument("--model_filename", type=str, help="Model filename.", default="./model.pth")
    args = parser.parse_args()

    filename = os.path.join(args.model_dir, args.filename)
    model_filepath = os.path.join(args.model_dir, args.model_filename)

    p = np.round(np.arange(0, 1.1, 0.1), 2)
    for i_p in p:
        print('RSH p value: ', i_p)
        data_transforms = {
            'train': transforms.Compose([
                # For CIFAR-10 and CIFAR100, either change the model or resize images to 64x64 (uncomment the transform below)
                # transforms.Resize(64),
                DiskAugmenter(local_mask=(120, 160), global_mask=(40, 80), augmenting_prob=0),
                RandomShadows(p=i_p, high_ratio=(1,2), low_ratio=(0,1), \
                left_low_ratio=(0.4,0.8), left_high_ratio=(0,0.3), right_low_ratio=(0.4,0.8),
                right_high_ratio = (0,0.3)), ## high means from top of image, low means from top to bottom low
                RandomGamma(gamma_p = 0, gamma_ratio=(0, 1.5)),
                RandomColorJitter(p = 0, brightness_ratio=(0,2), contrast_ratio=(0,2), \
                            saturation_ratio=(0,2), hue_ratio=(-0.5,0.5)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                # transforms.RandomErasing(p=i_p)
            ]),
            'val': transforms.Compose([
                # For CIFAR-10 and CIFAR100, either change the model or resize images to 64x64 (uncomment the transform below)
                # transforms.Resize(64),
                RandomShadows(p=1, high_ratio=(1,2), low_ratio=(0,1), \
                left_low_ratio=(0.4,0.8), left_high_ratio=(0,0.3), right_low_ratio=(0.4,0.8),
                right_high_ratio = (0,0.3)), ## high means from top of image, low means from top to bottom low
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
            ])
        }

        if args.dataset == "TinyImageNet":
            image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(args.data_dir, x),
                                                      data_transforms[x])
                              for x in ['train', 'val','test']}
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                         shuffle=True, num_workers=2)
                          for x in ['train', 'val', 'test']}
            dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
        else:
            # For CIFAR-10 and CIFAR100, either change the model or resize images to 64x64 (uncomment the transform above)
            if args.dataset == "CIFAR10":
                train_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=data_transforms['train'])
                test_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=data_transforms['val'])
            elif args.dataset == "CIFAR100":
                train_set = torchvision.datasets.CIFAR100(root='./CIFAR100', train=True, download=True, transform=data_transforms['train'])
                test_set = torchvision.datasets.CIFAR100(root='./CIFAR100', train=False, download=True, transform=data_transforms['val'])

            else:
                print("Only \"TinyImageNet\", \"CIFAR10\", \"CIFAR100\" datasets are supported.")
                exit()

            train_sampler = RandomSampler(train_set)
            test_sampler = SequentialSampler(test_set)

            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

            image_datasets = {'train': train_set, 'val': test_set}
            dataloaders = {'train': train_loader, 'val': test_loader}
            dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        if args.dataset == "TinyImageNet":
            classes = 200
        elif args.dataset == "CIFAR100":
            classes = 100
        elif args.dataset == "CIFAR10":
            classes = 10

        if args.model_name == "EfficientNet":
            model_ft = EfficientNet.from_pretrained('efficientnet-b0', num_classes=classes)
        elif args.model_name == "AlexNet":
            model_ft = alexnet(pretrained=True, out_classes=classes)
        else:
            print("Only \"EfficientNet\" or \"AlexNet\" models are supported.")
            exit()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_ft = model_ft.to(device)
        #Loss Function
        criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.learning_rate, momentum=0.9)

        #Train
        train_model(args.model_name,model_ft, dataloaders, dataset_sizes, \
        criterion, optimizer_ft, i_p, args.model_dir, \
        filename, args.num_epochs)
