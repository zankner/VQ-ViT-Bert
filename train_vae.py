import argparse
from vae import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VQ-VAE')

    # Directory args
    parser.add_argument("--data_dir",
                        default="./data",
                        help="location dataset is stored",
                        type=str)
    parser.add_argument("--checkpoint_dir", default="./vae_ckpt", type=str)
    parser.add_argument("--summary_dir", default="./vae_runs", type=str)

    # Training args
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--val_size", default=0.15, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--step_size", default=30, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--print_freq", default=50, type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_workers", default=3, type=int)

    # Vae args
    parser.add_argument("--num_codebook_indeces", default=1024, type=int)
    parser.add_argument("--embedding_dim", default=256, type=int)
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--feature_dim", default=64, type=int)
    parser.add_argument("--channels", default=3, type=int)
    parser.add_argument("--commitment_cost", default=0.6, type=int)

    args = parser.parse_args()

    train(args)