import os
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torchvision import transforms, datasets
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import datetime
from vae import VQVae
from vae.model_utils import train_step, validate_step
from utils import get_train_val_loaders, get_test_loader

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _show(img):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()


def _unormalize(tensor):
    for t, m, s in zip(tensor, MEAN, STD):
        t.mul_(s).add_(m)


def visualize(args):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    args.batch_size = args.num_examples
    test_loader = get_test_loader(args)

    model = VQVae(args.num_codebook_indeces, args.embedding_dim,
                  args.num_blocks, args.feature_dim, args.channels,
                  args.commitment_cost)

    state_dict = torch.load(args.ckpt_path)["model_state_dict"]
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    (data, _) = next(iter(test_loader))
    data = data.to(device)

    recon, _, _ = model(data)
    print(recon.shape)

    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(MEAN, STD)], std=[1 / s for s in STD])

    data = inv_normalize(data)
    recon = inv_normalize(recon)

    _show(make_grid(data.cpu().data))
    _show(make_grid(recon.cpu().data))
