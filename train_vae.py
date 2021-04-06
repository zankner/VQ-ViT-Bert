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
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--val_size", type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--step_size", default=30, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--print_freq", default=50, type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    # Vae args
    parser.add_argument("--num_codebook_indeces", default=1000, type=int)
    parser.add_argument("--embedding_dim", default=512, type=int)
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--feature_dim", default=128, type=int)
    parser.add_argument("--channels", default=3, type=int)
    parser.add_argument("--commitment_cost", default=0.5, type=float)

    args = parser.parse_args()

    train(args)