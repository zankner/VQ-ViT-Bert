import os
import argparse
import datetime
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from vae import VQVae, OpenAIDiscreteVAE, VQGanVAE1024
from utils import get_train_val_loaders


def token_dist(args):
    device = torch.device("cuda:0")

    train_loader, val_loader = get_train_val_loaders(args)

    log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.architecture == "dall-e":
        vae = OpenAIDiscreteVAE()
    elif args.architecture == "vq-gan":
        vae = VQGanVAE1024()
    else:
        assert args.vae_ckpt != None, "Vae checkpoint must be specified when loading a custom trained VAE"

        print("Loading pretrained VQ-VAE .....")

        vae = VQVae(args.num_codebook_indeces, args.embedding_dim,
                    args.num_blocks, args.feature_dim, args.channels)
        vae_ckpt = torch.load(args.vae_ckpt)['model_state_dict']
        vae.load_state_dict(vae_ckpt)
    vae.to(device)

    print("Generating dist train set")
    train_dist_dict = {code: 0 for code in range(args.num_codebook_indeces)}
    for (image, _) in tqdm(train_loader):
        image = image.cuda()

        tokens = vae.get_codebook_indices(image).cpu().numpy()
        for image_tokens in tokens:
            for token in image_tokens:
                train_dist_dict[token] += 1

    plt.bar(list(train_dist_dict.keys()), train_dist_dict.values())
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize tokens')

    # Directory args
    parser.add_argument("--data_dir",
                        default="./data/ImageNet",
                        help="location dataset is stored",
                        type=str)
    parser.add_argument("--vae_ckpt", type=str)

    # Training args
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--val_size", type=float)
    parser.add_argument("--random_state", type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--dataset", default="image-net-tar", type=str)

    # Vae args
    parser.add_argument("--architecture", type=str)
    parser.add_argument("--num_codebook_indeces", default=1000, type=int)
    parser.add_argument("--embedding_dim", default=512, type=int)
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--feature_dim", default=128, type=int)
    parser.add_argument("--channels", default=3, type=int)
    parser.add_argument("--commitment_cost", default=0.5, type=float)

    args = parser.parse_args()

    token_dist(args)
