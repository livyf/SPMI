import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .wideResnet import WideResNet
from .utils_algo import *
from .cutout import Cutout
from .autoaugment import SVHNPolicy


mean = [0.4380, 0.4440, 0.4730]
std = [0.1751, 0.1771, 0.1744]

def load_svhn(args):
    # original training dataset and labels
    original_train = datasets.SVHN(root=args.data_dir, split='train', transform=transforms.ToTensor(), download=True)
    original_train.labels = torch.tensor(original_train.labels)

    # testing dataset and dataloader
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
    ])
    test_dataset = datasets.SVHN(root=args.data_dir, split='test', transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size*4,
        pin_memory=True, shuffle=False, num_workers=args.workers)

    # split dataset into labeled and unlabeled
    print('labeled num: ', args.labeled_num)

    if args.labeled_num == len(original_train.labels):
        labeled_idx, unlabeled_idx = range(len(original_train.labels)), range(len(original_train.labels))
    else:
        labeled_idx, unlabeled_idx = x_u_split(args, original_train.labels)
        assert torch.sum(torch.bincount(original_train.labels[labeled_idx]) == args.labeled_num // args.num_class) == args.num_class

    ori_data = original_train.data[labeled_idx]
    ori_labels = original_train.labels[labeled_idx].long()
    unlabeled_data = original_train.data[unlabeled_idx]
    unlabeled_gt = original_train.labels[unlabeled_idx]

    if args.exp_type == 'rand':
        partialY_matrix = generate_uniform_cv_candidate_labels(args, ori_labels)
    elif args.exp_type == 'ins':
        ori_data = torch.Tensor(original_train.data)
        model = WideResNet(depth=16, num_classes=10, widen_factor=8, dropRate=0.4)
        model.load_state_dict(torch.load('./pmodel/svhn.pt'))
        partialY_matrix = generate_instancedependent_candidate_labels(model, ori_data, ori_labels)
        ori_data = original_train.data[labeled_idx]

    temp = torch.zeros(partialY_matrix.shape)
    temp[torch.arange(partialY_matrix.shape[0]), ori_labels] = 1
    
    if torch.sum(partialY_matrix * temp) == partialY_matrix.shape[0]:
        print('Partial labels correctly loaded !')
    else:
        print('Inconsistent permutation !')

    print('Average candidate num: ', partialY_matrix.sum(1).mean())
    
    # generate partial label dataset
    partial_training_dataset = SVHN_Partialize(ori_data, partialY_matrix.float(), ori_labels.float())
    partial_training_dataloader = torch.utils.data.DataLoader(
        dataset=partial_training_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        # drop_last=True
    )

    # unlabeled training
    if len(unlabeled_data) == 0:
        return partial_training_dataloader, partialY_matrix, None, test_loader
    unlabeled_training_dataset = SVHN_Unlabeled(unlabeled_data, unlabeled_gt)
    unlabeled_training_dataloader = torch.utils.data.DataLoader(
        dataset=unlabeled_training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        # drop_last=True
    )

    return partial_training_dataloader, partialY_matrix, unlabeled_training_dataloader, test_loader


def x_u_split(args, labels):
    labeled_num = args.labeled_num
    label_per_class = labeled_num // args.num_class
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_class):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == labeled_num

    return labeled_idx, unlabeled_idx


class SVHN_Partialize(Dataset):
    def __init__(self, images, given_partial_label_matrix, true_labels):
        
        self.ori_images = np.transpose(images, (0, 2, 3, 1))
        self.given_partial_label_matrix = given_partial_label_matrix
        self.pseudo_cand_labels = given_partial_label_matrix
        self.true_labels = true_labels
        self.pre = False

        self.ori_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.weak_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.strong_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),

            SVHNPolicy(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=8),
            
            transforms.Normalize(mean, std)
        ])

    def __len__(self):
        return len(self.true_labels)

    def set_pseudo_cand_labels(self, batch_pseudo_cand_labels, index, ema_theta):
        batch_pseudo_cand_labels = batch_pseudo_cand_labels.cpu()
        index = index.cpu()
        self.pseudo_cand_labels[index] = self.pseudo_cand_labels[index] * ema_theta \
                                         + batch_pseudo_cand_labels * (1-ema_theta)

    def __getitem__(self, index):
        each_label = self.pseudo_cand_labels[index]
        each_true_label = self.true_labels[index]
        if self.pre:
            image = self.weak_transform(self.ori_images[index])
            each_ori_label = self.given_partial_label_matrix[index]
            return image, each_label, each_ori_label, each_true_label, index
        else:
            # image = self.ori_transform(self.ori_images[index])
            # image = self.weak_transform(self.ori_images[index])
            image = self.strong_transform(self.ori_images[index])
            return image, each_label, each_true_label, index


class SVHN_Unlabeled(Dataset):
    def __init__(self, images, true_labels):
        self.ori_images = np.transpose(images, (0, 2, 3, 1))
        self.pseudo_complementary_labels = torch.zeros(len(true_labels), 10).cuda()
        self.outputs = nn.functional.normalize(torch.randn((len(true_labels), 10)), dim=0)
        self.posterior = (torch.ones(10)/10).to(torch.float).cuda()
        self.true_labels = true_labels
        self.pre = False

        self.ori_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.weak_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.strong_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),

            SVHNPolicy(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=8),

            transforms.Normalize(mean, std)
        ])

    def __len__(self):
        return len(self.true_labels)

    def set_pseudo_complementary_labels(self, batch_pseudo_complementary_labels, index, init=False):
        self.pseudo_complementary_labels[index] = batch_pseudo_complementary_labels
        # if init:
        #     self.pseudo_complementary_labels[index] = batch_pseudo_complementary_labels
        # else:
        #     # EMA
        #     self.pseudo_complementary_labels[index] = self.pseudo_complementary_labels[index] * 0.9 + \
        #                                               batch_pseudo_complementary_labels * 0.1

    def set_posterior(self, posterior):
        self.posterior = posterior

    def __getitem__(self, index):
        if self.pre:
            image = self.weak_transform(self.ori_images[index])
        else:
            image = self.strong_transform(self.ori_images[index])
        each_comp_label = self.pseudo_complementary_labels[index].cpu()
        each_true_label = self.true_labels[index]
        return image, each_comp_label, each_true_label, index