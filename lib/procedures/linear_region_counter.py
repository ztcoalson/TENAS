import sys
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
from pdb import set_trace as bp
from datasets import CUTOUT, Dataset2Class, ImageNet16
from operator import mul
from functools import reduce

from torch.utils.data import Dataset

sys.path.append("../poisons/")
from poisons import LabelFlippingPoisoningDataset, CleanLabelPoisoningDataset
from poisons_utils import imshow

# ------------------------------------------------------------------------------
#   Diffusion Denoise Stuff
# ------------------------------------------------------------------------------
class DenoisedDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.pdata     = data
        self.plabels   = labels
        self.transform = transform
        print ('DenoisedDataset: read the data - {}'.format(self.pdata.shape))


    def __len__(self):
        # return the number of instances in a dataset
        return len(self.pdata)


    def __getitem__(self, idx):
        # return (data, label) where label is index of the label class
        data, label = self.pdata[idx], self.plabels[idx]

        # transform into the Pytorch format
        if self.transform:
            data = self.transform(data)
        return (data, label)

def load_denoised_cifar(datpath=None, train_transform=None, valid_transform=None):
    # extract the data
    custom_dataset = torch.load(datpath)

    # compose the dataset
    trainset = DenoisedDataset(custom_dataset['tdata'],
                               custom_dataset['tlabels'],
                               transform=train_transform)
    validset = DenoisedDataset(custom_dataset['vdata'],
                               custom_dataset['vlabels'],
                               transform=valid_transform)
    return trainset, validset

class RandChannel(object):
    # randomly pick channels from input
    def __init__(self, num_channel):
        self.num_channel = num_channel

    def __repr__(self):
        return ('{name}(num_channel={num_channel})'.format(name=self.__class__.__name__, **self.__dict__))

    def __call__(self, img):
        channel = img.size(0)
        channel_choice = sorted(np.random.choice(list(range(channel)), size=self.num_channel, replace=False))
        return torch.index_select(img, 0, torch.Tensor(channel_choice).long())


def get_datasets(name, root, input_size, cutout=-1, poisons_type="none", poisons_path=None):
    assert len(input_size) in [3, 4]
    if len(input_size) == 4:
        input_size = input_size[1:]
    assert input_size[1] == input_size[2]

    if name == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std  = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif name == 'mnist':
        mean = [0.1307] * 3
        std  = [0.3081] * 3
    elif name == 'svhn':
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
    elif name == 'fashion_mnist':
        mean = [0.2856] * 3
        std = [0.3385] * 3
    elif name == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std  = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif name.startswith('imagenet-1k'):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif name.startswith('ImageNet16'):
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std  = [x / 255 for x in [63.22,  61.26 , 65.09]]
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    # Data Argumentation
    if name == 'cifar10' or name == 'cifar100':
        lists = [transforms.RandomCrop(input_size[1], padding=0), transforms.ToTensor(), transforms.Normalize(mean, std), RandChannel(input_size[0])]
        if cutout > 0 : lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    elif name == 'mnist':
        lists = [transforms.Resize((32, 32)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize(mean, std), RandChannel(input_size[0])]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    elif name == 'svhn':
        lists = [transforms.ToTensor(), transforms.Normalize(mean, std), RandChannel(input_size[0])]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    elif name == 'fashion_mnist':
        lists = [transforms.Resize((32, 32)), transforms.Grayscale(num_output_channels=3), transforms.RandomCrop(input_size[1], padding=0), transforms.ToTensor(), transforms.Normalize(mean, std), RandChannel(input_size[0])]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    elif name.startswith('ImageNet16'):
        lists = [transforms.RandomCrop(input_size[1], padding=0), transforms.ToTensor(), transforms.Normalize(mean, std), RandChannel(input_size[0])]
        if cutout > 0 : lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    elif name.startswith('imagenet-1k'):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if name == 'imagenet-1k':
            xlists    = []
            xlists.append(transforms.Resize((32, 32), interpolation=2))
            xlists.append(transforms.RandomCrop(input_size[1], padding=0))
        elif name == 'imagenet-1k-s':
            xlists = [transforms.RandomResizedCrop(32, scale=(0.2, 1.0))]
            xlists = []
        else: raise ValueError('invalid name : {:}'.format(name))
        xlists.append(transforms.ToTensor())
        xlists.append(normalize)
        xlists.append(RandChannel(input_size[0]))
        train_transform = transforms.Compose(xlists)
        test_transform = transforms.Compose([transforms.Resize(40), transforms.CenterCrop(32), transforms.ToTensor(), normalize])
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    if name == 'cifar10':
        if poisons_type == "none":
            train_data = dset.CIFAR10 (root, train=True, transform=train_transform, download=True)
        else:
            train_kwargs = {
                'root': root,
                'train': True,
                'download': True,
                'transform': None
            }

            if poisons_type == "label_flip":
                train_data = LabelFlippingPoisoningDataset(poisons_path, train_transform, train_kwargs)
                print("Added {} label-flip poisons to CIFAR10".format(train_data.get_num_poisons()))
            elif poisons_type == "clean_label":
                train_data = CleanLabelPoisoningDataset(poisons_path, train_transform, train_kwargs)
                print("Added {} clean-label poisons to CIFAR10".format(train_data.get_num_poisons()))
            elif poisons_type == 'diffusion_denoise':
                # create new transforms without ToTensor)
                lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.Normalize(mean, std), RandChannel(input_size[0])]
                if cutout > 0 : lists += [CUTOUT(cutout)]
                train_transform = transforms.Compose(lists)
                test_transform  = transforms.Compose([transforms.Normalize(mean, std)])
                train_data, test_data = load_denoised_cifar(poisons_path, train_transform, test_transform)
                print("Loaded diffusion denoised CIFAR10")
            else:
                raise ValueError("Invalid poisons_type: {}".format(poisons_type))

        if poisons_type != 'diffusion_denoise':
            test_data = dset.CIFAR10 (root, train=False, transform=test_transform , download=True)

        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name == 'mnist':
        train_data = dset.MNIST(root, train=True , transform=train_transform, download=True)
        test_data  = dset.MNIST(root, train=False, transform=test_transform , download=True)
        assert len(train_data) == 60000 and len(test_data) == 10000
    elif name == 'svhn':
        train_data = dset.SVHN(root, split='train', transform=train_transform, download=True)
        test_data  = dset.SVHN(root, split='test',  transform=test_transform , download=True)
        assert len(train_data) == 73257 and len(test_data) == 26032
    elif name == 'fashion_mnist':
        train_data = dset.FashionMNIST(root, train=True , transform=train_transform, download=True)
        test_data  = dset.FashionMNIST(root, train=False, transform=test_transform , download=True)
        assert len(train_data) == 60000 and len(test_data) == 10000
    elif name == 'cifar100':
        train_data = dset.CIFAR100(root, train=True , transform=train_transform, download=True)
        test_data  = dset.CIFAR100(root, train=False, transform=test_transform , download=True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name.startswith('imagenet-1k'):
        train_data = dset.ImageFolder(osp.join(root, 'train'), train_transform)
        test_data  = dset.ImageFolder(osp.join(root, 'val'),   test_transform)
    elif name == 'ImageNet16':
        train_data = ImageNet16(root, True , train_transform)
        test_data  = ImageNet16(root, False, test_transform)
        assert len(train_data) == 1281167 and len(test_data) == 50000
    elif name == 'ImageNet16-120':
        train_data = ImageNet16(root, True , train_transform, 120)
        test_data  = ImageNet16(root, False, test_transform , 120)
        assert len(train_data) == 151700 and len(test_data) == 6000
    elif name == 'ImageNet16-150':
        train_data = ImageNet16(root, True , train_transform, 150)
        test_data  = ImageNet16(root, False, test_transform , 150)
        assert len(train_data) == 190272 and len(test_data) == 7500
    elif name == 'ImageNet16-200':
        train_data = ImageNet16(root, True , train_transform, 200)
        test_data  = ImageNet16(root, False, test_transform , 200)
        assert len(train_data) == 254775 and len(test_data) == 10000
    else: raise TypeError("Unknow dataset : {:}".format(name))

    class_num = Dataset2Class[name]
    return train_data, test_data, class_num


class LinearRegionCount(object):
    """Computes and stores the average and current value"""
    def __init__(self, n_samples):
        self.ActPattern = {}
        self.n_LR = -1
        self.n_samples = n_samples
        self.ptr = 0
        self.activations = None

    @torch.no_grad()
    def update2D(self, activations):
        n_batch = activations.size()[0]
        n_neuron = activations.size()[1]
        self.n_neuron = n_neuron
        if self.activations is None:
            self.activations = torch.zeros(self.n_samples, n_neuron).cuda()
        self.activations[self.ptr:self.ptr+n_batch] = torch.sign(activations)  # after ReLU
        self.ptr += n_batch

    @torch.no_grad()
    def calc_LR(self):
        res = torch.matmul(self.activations.half(), (1-self.activations).T.half()) # each element in res: A * (1 - B)
        res += res.T # make symmetric, each element in res: A * (1 - B) + (1 - A) * B, a non-zero element indicate a pair of two different linear regions
        res = 1 - torch.sign(res) # a non-zero element now indicate two linear regions are identical
        res = res.sum(1) # for each sample's linear region: how many identical regions from other samples
        res = 1. / res.float() # contribution of each redudant (repeated) linear region
        self.n_LR = res.sum().item() # sum of unique regions (by aggregating contribution of all regions)
        del self.activations, res
        self.activations = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def update1D(self, activationList):
        code_string = ''
        for key, value in activationList.items():
            n_neuron = value.size()[0]
            for i in range(n_neuron):
                if value[i] > 0:
                    code_string += '1'
                else:
                    code_string += '0'
        if code_string not in self.ActPattern:
            self.ActPattern[code_string] = 1

    def getLinearReginCount(self):
        if self.n_LR == -1:
            self.calc_LR()
        return self.n_LR


class Linear_Region_Collector:
    def __init__(self, models=[], input_size=(64, 3, 32, 32), sample_batch=100, dataset='cifar100', data_path=None, seed=0, poisons_type="none", poisons_path=None):
        self.models = []
        self.input_size = input_size  # BCHW
        self.sample_batch = sample_batch
        self.input_numel = reduce(mul, self.input_size, 1)
        self.interFeature = []
        self.dataset = dataset
        self.data_path = data_path
        self.seed = seed
        self.reinit(models, input_size, sample_batch, seed, poisons_type, poisons_path)

    def reinit(self, models=None, input_size=None, sample_batch=None, seed=None, poisons_type="none", poisons_path=None):
        if models is not None:
            assert isinstance(models, list)
            del self.models
            self.models = models
            for model in self.models:
                self.register_hook(model)
            self.LRCounts = [LinearRegionCount(self.input_size[0]*self.sample_batch) for _ in range(len(models))]
        if input_size is not None or sample_batch is not None:
            if input_size is not None:
                self.input_size = input_size  # BCHW
                self.input_numel = reduce(mul, self.input_size, 1)
            if sample_batch is not None:
                self.sample_batch = sample_batch
            if self.data_path is not None:
                self.train_data, _, class_num = get_datasets(self.dataset, self.data_path, self.input_size, -1, poisons_type, poisons_path)
                self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.input_size[0], num_workers=16, pin_memory=True, drop_last=True, shuffle=True)
                self.loader = iter(self.train_loader)
        if seed is not None and seed != self.seed:
            self.seed = seed
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def clear(self):
        self.LRCounts = [LinearRegionCount(self.input_size[0]*self.sample_batch) for _ in range(len(self.models))]
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def register_hook(self, model):
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                m.register_forward_hook(hook=self.hook_in_forward)

    def hook_in_forward(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            self.interFeature.append(output.detach())  # for ReLU

    def forward_batch_sample(self):
        for _ in range(self.sample_batch):
            try:
                inputs, targets = self.loader.next()
            except Exception:
                del self.loader
                self.loader = iter(self.train_loader)
                inputs, targets = self.loader.next()
            for model, LRCount in zip(self.models, self.LRCounts):
                self.forward(model, LRCount, inputs)
        return [LRCount.getLinearReginCount() for LRCount in self.LRCounts]

    def forward(self, model, LRCount, input_data):
        self.interFeature = []
        with torch.no_grad():
            model.forward(input_data.cuda())
            if len(self.interFeature) == 0: return
            feature_data = torch.cat([f.view(input_data.size(0), -1) for f in self.interFeature], 1)
            LRCount.update2D(feature_data)
