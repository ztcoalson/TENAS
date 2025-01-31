import os, sys, time, argparse
import math
import random
from easydict import EasyDict as edict
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from pathlib import Path
lib_dir = (Path(__file__).parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from datasets import get_datasets, get_nas_search_loaders
from procedures import prepare_seed, prepare_logger
from procedures import Linear_Region_Collector, get_ntk_n
from utils import get_model_infos
from log_utils import time_string
from models import get_cell_based_tiny_net, get_search_spaces  # , nas_super_nets
from nas_201_api import NASBench201API as API
from pdb import set_trace as bp

import json

import torchvision.datasets as dset
from torch.utils.data import Subset

from torchvision import transforms

sys.path.append("../poisons/")
from poisons import LabelFlippingPoisoningDataset, CleanLabelPoisoningDataset
from poisons_utils import imshow

INF = 1000  # used to mark prunned operators


def kaiming_normal_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def kaiming_normal_fanout_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def init_model(model, method='kaiming_norm_fanin'):
    if method == 'kaiming_norm_fanin':
        model.apply(kaiming_normal_fanin_init)
    elif method == 'kaiming_norm_fanout':
        model.apply(kaiming_normal_fanout_init)
    return model


def round_to(number, precision, eps=1e-8):
    # round to significant figure
    dtype = type(number)
    if number == 0:
        return number
    sign = number / abs(number)
    number = abs(number) + eps
    power = math.floor(math.log(number, 10)) + 1
    if dtype == int:
        return int(sign * round(number*10**(-power), precision) * 10**(power))
    else:
        return sign * round(number*10**(-power), precision) * 10**(power)

def get_ntk_one_batch(inputs, targets, networks, recalbn=0, train_mode=False):
    device = torch.cuda.current_device()
    # if recalbn > 0:
    #     network = recal_bn(network, xloader, recalbn, device)
    #     if network_2 is not None:
    #         network_2 = recal_bn(network_2, xloader, recalbn, device)
    ntks = []
    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()
    ######
    grads = [[] for _ in range(len(networks))]

    inputs = inputs.cuda(device=device, non_blocking=True)
    for net_idx, network in enumerate(networks):
        network.zero_grad()
        inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
        logit = network(inputs_)
        if isinstance(logit, tuple):
            logit = logit[1]  # 201 networks: return features and logits
        for _idx in range(len(inputs_)):
            logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
            grad = []
            for name, W in network.named_parameters():
                if 'weight' in name and W.grad is not None:
                    grad.append(W.grad.view(-1).detach())
            grads[net_idx].append(torch.cat(grad, -1))
            network.zero_grad()
            torch.cuda.empty_cache()
    ######
    grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
    conds = []
    for ntk in ntks:
        eigenvalues, _ = torch.symeig(ntk)  # ascending
        conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))
    return conds

def data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform

def build_thin_network(xargs, model_config_thin):
    network_thin = get_cell_based_tiny_net(model_config_thin).cuda().train()
    init_model(network_thin, xargs.init)
    
    arch_parameters = [alpha.detach().clone() for alpha in network_thin.get_alphas()]
    for alpha in arch_parameters:
        alpha[:, :] = 0
    
    for alpha in arch_parameters:
        alpha[:, 0] = -INF
    network_thin.set_alphas(arch_parameters)

    return network_thin

def build_network(xargs, model_config):
    network = get_cell_based_tiny_net(model_config).cuda().train()
    init_model(network, xargs.init+"_fanout" if xargs.init.startswith('kaiming') else xargs.init)
    
    arch_parameters = [alpha.detach().clone() for alpha in network.get_alphas()]
    for alpha in arch_parameters:
        alpha[:, :] = 0
    
    for alpha in arch_parameters:
        alpha[:, 0] = -INF
    network.set_alphas(arch_parameters)

    return network

def sample_regions(xargs, network_thin, lrc_model, lrc_queue):
    lrc_model.reinit(models=[network_thin], seed=xargs.rand_seed)
    lrc_model.train_loader = lrc_queue
    lrc_model.loader = iter(lrc_queue)

    regions = lrc_model.forward_batch_sample()[0]
    lrc_model.clear()

    return regions

def sample_ntk(network, loader):
    total_ntk = 0
    for inputs, targets in loader:
        ntk = get_ntk_one_batch(inputs, targets, [network], recalbn=0, train_mode=True)[0]
        total_ntk += ntk
    
    return total_ntk / len(loader)

def percentage_change(old, new):
    return ((new - old) / abs(old)) * 100

def main(xargs):
    PID = os.getpid()
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    prepare_seed(xargs.rand_seed)

    search_space = get_search_spaces('cell', xargs.search_space_name)
    model_config = edict({'name': 'DARTS-V1',
                            'C': 1, 'N': 1, 'depth': 2, 'use_stem': True, 'stem_multiplier': 1,
                            'num_classes': 10,
                            'space': search_space,
                            'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                            'super_type': xargs.super_type,
                            'steps': 4,
                            'multiplier': 4,
                            })
    model_config_thin = edict({'name': 'DARTS-V1',
                                'C': 1, 'N': 1, 'depth': 2, 'use_stem': False, 'stem_multiplier': 1,
                                'max_nodes': xargs.max_nodes, 'num_classes': 10,
                                'space': search_space,
                                'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                                'super_type': xargs.super_type,
                                'steps': 4,
                                'multiplier': 4,
                                })
    
    # build data loaders
    train_transform, valid_transform = data_transforms_cifar10()    

    # for poisoning attacks
    train_kwargs = {
        'root': args.data_path,
        'train': True,
        'download': True,
        'transform': None,
    }

    indices = np.random.choice(50000, 100, replace=False)

    clean_train_data = dset.CIFAR10(root=xargs.data_path, train=True, download=True, transform=valid_transform)
    clean_data = Subset(clean_train_data, indices)
    clean_queue = torch.utils.data.DataLoader(
        clean_data, batch_size=xargs.batch_size,
        shuffle=False,
        pin_memory=True, num_workers=4
    )

    rlf_train_data = LabelFlippingPoisoningDataset("/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/poisons/poisons/rlf/rlf-cifar10-50.0%.pth", valid_transform, train_kwargs)
    rlf_data = Subset(rlf_train_data, indices)
    rlf_queue = torch.utils.data.DataLoader(
        rlf_data, batch_size=xargs.batch_size,
        shuffle=False,
        pin_memory=True, num_workers=4
    )
    
    clf_train_data = LabelFlippingPoisoningDataset("/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/poisons/poisons/smart-lf/smart-lf-resnet18-cifar10-50.0%.pth", valid_transform, train_kwargs)
    clf_data = Subset(clf_train_data, indices)
    clf_queue = torch.utils.data.DataLoader(
        clf_data, batch_size=xargs.batch_size,
        shuffle=False,
        pin_memory=True, num_workers=4
    )
    
    noise_train_data = CleanLabelPoisoningDataset("/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/poisons/poisons/noise/noise-cifar10-50.0%.pth", valid_transform, train_kwargs)
    noise_data = Subset(noise_train_data, indices)
    noise_queue = torch.utils.data.DataLoader(
        noise_data, batch_size=xargs.batch_size,
        shuffle=False,
        pin_memory=True, num_workers=4
    )
    
    gc_train_data = CleanLabelPoisoningDataset("/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/poisons/gc_runs/te-nas/gc/tenas-eps=0.5-gc-d-darts-50.0%-20250116-192302/poisons.pth", valid_transform, train_kwargs)
    gc_data = Subset(gc_train_data, indices)
    gc_queue = torch.utils.data.DataLoader(
        gc_data, batch_size=xargs.batch_size,
        shuffle=False,
        pin_memory=True, num_workers=4
    )

    lrc_clean_model = Linear_Region_Collector(
        input_size=(1000, 1, 3, 3), 
        sample_batch=1, 
        dataset=xargs.dataset, 
        data_path=xargs.data_path, 
        seed=xargs.rand_seed, 
        poisons_type='none', 
        poisons_path=''
    )
    lrc_clean_queue = torch.utils.data.DataLoader(
        Subset(lrc_clean_model.train_data, indices), 
        batch_size=1000,
        shuffle=False,
        pin_memory=True, num_workers=4
    )

    lrc_rlf_model = Linear_Region_Collector(
        input_size=(1000, 1, 3, 3), 
        sample_batch=1, 
        dataset=xargs.dataset, 
        data_path=xargs.data_path, 
        seed=xargs.rand_seed, 
        poisons_type='label_flip', 
        poisons_path='/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/poisons/poisons/rlf/rlf-cifar10-50.0%.pth'
    )
    lrc_rlf_queue = torch.utils.data.DataLoader(
        Subset(lrc_rlf_model.train_data, indices), 
        batch_size=1000,
        shuffle=False,
        pin_memory=True, num_workers=4
    )

    lrc_clf_model = Linear_Region_Collector(
        input_size=(1000, 1, 3, 3), 
        sample_batch=1, 
        dataset=xargs.dataset, 
        data_path=xargs.data_path, 
        seed=xargs.rand_seed, 
        poisons_type='label_flip', 
        poisons_path='/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/poisons/poisons/smart-lf/smart-lf-resnet18-cifar10-50.0%.pth'
    )
    lrc_clf_queue = torch.utils.data.DataLoader(
        Subset(lrc_clf_model.train_data, indices), 
        batch_size=1000,
        shuffle=False,
        pin_memory=True, num_workers=4
    )

    lrc_noise_model = Linear_Region_Collector(
        input_size=(1000, 1, 3, 3), 
        sample_batch=1, 
        dataset=xargs.dataset, 
        data_path=xargs.data_path, 
        seed=xargs.rand_seed, 
        poisons_type='clean_label', 
        poisons_path='/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/poisons/poisons/noise/noise-cifar10-50.0%.pth'
    )
    lrc_noise_queue = torch.utils.data.DataLoader(
        Subset(lrc_noise_model.train_data, indices), 
        batch_size=1000,
        shuffle=False,
        pin_memory=True, num_workers=4
    )

    lrc_gc_model = Linear_Region_Collector(
        input_size=(1000, 1, 3, 3), 
        sample_batch=1, 
        dataset=xargs.dataset, 
        data_path=xargs.data_path, 
        seed=xargs.rand_seed, 
        poisons_type='clean_label', 
        poisons_path='/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/poisons/gc_runs/te-nas/gc/tenas-eps=0.5-gc-d-darts-50.0%-20250116-192302/poisons.pth'
    )
    lrc_gc_queue = torch.utils.data.DataLoader(
        Subset(lrc_gc_model.train_data, indices), 
        batch_size=1000,
        shuffle=False,
        pin_memory=True, num_workers=4
    )

    results = {
        'clean': {},
        'rlf': {},
        'clf': {},
        'noise': {},
        'gc': {}
    }

    for _ in tqdm(range(xargs.n_sample), desc='Sample NTK and Regions', total=xargs.n_sample):
        network = build_network(xargs, model_config)
        network_thin = build_thin_network(xargs, model_config_thin)

        clean_ntk = sample_ntk(network, clean_queue)
        clean_regions = sample_regions(xargs, network_thin, lrc_clean_model, lrc_clean_queue)

        results['clean']['ntk'] = results['clean'].get('ntk', []) + [clean_ntk]
        results['clean']['regions'] = results['clean'].get('regions', []) + [clean_regions]

        rlf_ntk = sample_ntk(network, rlf_queue)
        rlf_regions = sample_regions(xargs, network_thin, lrc_rlf_model, lrc_rlf_queue)

        results['rlf']['ntk'] = results['rlf'].get('ntk', []) + [percentage_change(clean_ntk, rlf_ntk)]
        results['rlf']['regions'] = results['rlf'].get('regions', []) + [percentage_change(clean_regions, rlf_regions)]

        clf_ntk = sample_ntk(network, clf_queue)
        clf_regions = sample_regions(xargs, network_thin, lrc_clean_model, lrc_clf_queue)

        results['clf']['ntk'] = results['clf'].get('ntk', []) + [percentage_change(clean_ntk, clf_ntk)]
        results['clf']['regions'] = results['clf'].get('regions', []) + [percentage_change(clean_regions, clf_regions)]

        noise_ntk = sample_ntk(network, noise_queue)
        noise_regions = sample_regions(xargs, network_thin, lrc_noise_model, lrc_noise_queue)

        results['noise']['ntk'] = results['noise'].get('ntk', []) + [percentage_change(clean_ntk, noise_ntk)]
        results['noise']['regions'] = results['noise'].get('regions', []) + [percentage_change(clean_regions, noise_regions)]

        gc_ntk = sample_ntk(network, gc_queue)
        gc_regions = sample_regions(xargs, network_thin, lrc_gc_model, lrc_gc_queue)

        results['gc']['ntk'] = results['gc'].get('ntk', []) + [percentage_change(clean_ntk, gc_ntk)]
        results['gc']['regions'] = results['gc'].get('regions', []) + [percentage_change(clean_regions, gc_regions)]

    for key in results.keys():
        ntk_values = np.array(results[key]['ntk'])
        ntk_values = ntk_values[np.isfinite(ntk_values)]

        if len(ntk_values) > 0:
            ntk_mean = np.mean(ntk_values)
            ntk_std = np.std(ntk_values)
        else:
            ntk_mean = np.nan
            ntk_std = np.nan
        
        results[key]['ntk'] = {
            'mean': ntk_mean,
            'std': ntk_std
        }

        regions_values = np.array(results[key]['regions'])
        regions_values = regions_values[np.isfinite(regions_values)]

        if len(regions_values) > 0:
            regions_mean = np.mean(regions_values)
            regions_std = np.std(regions_values)
        else:
            regions_mean = np.nan
            regions_std = np.nan

        results[key]['regions'] = {
            'mean': regions_mean,
            'std': regions_std
        }

    with open(f"./metrics_{xargs.n_sample}_{xargs.note}.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("TENAS")
    parser.add_argument('--data_path', type=str, help='Path to dataset')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'ImageNet16-120', 'imagenet-1k', 'mnist'], help='Choose between cifar10/100/ImageNet16-120/imagenet-1k')
    parser.add_argument('--search_space_name', type=str, default='nas-bench-201',  help='space of operator candidates: nas-bench-201 or darts.')
    parser.add_argument('--max_nodes', type=int, help='The maximum number of nodes.')
    parser.add_argument('--track_running_stats', type=int, choices=[0, 1], help='Whether use track_running_stats or not in the BN layer.')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 0)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for ntk')
    parser.add_argument('--save_dir', type=str, help='Folder to save checkpoints and log.')
    parser.add_argument('--arch_nas_dataset', type=str, help='The path to load the nas-bench-201 architecture dataset (tiny-nas-benchmark).')
    parser.add_argument('--rand_seed', type=int, help='manual seed')
    parser.add_argument('--precision', type=int, default=3, help='precision for % of changes of cond(NTK) and #Regions')
    parser.add_argument('--prune_number', type=int, default=1, help='number of operator to prune on each edge per round')
    parser.add_argument('--repeat', type=int, default=3, help='repeat calculation of NTK and Regions')
    parser.add_argument('--timestamp', default='none', type=str, help='timestamp for logging naming')
    parser.add_argument('--init', default='kaiming_normal', help='use gaussian init')
    parser.add_argument('--super_type', type=str, default='nasnet-super',  help='type of supernet: basic or nasnet-super')
    parser.add_argument('--n_sample', type=int, default=100, help='number of models to test')
    parser.add_argument('--note', type=str, default='', help='note to add to save file')

    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    main(args)
