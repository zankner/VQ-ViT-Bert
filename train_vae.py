import argparse
from vae import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VQ-VAE')

    parser.add_argument("--data_dir",
                        default="./data",
                        help="location dataset is stored",
                        type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--step_size", default=30, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--print_freq", default=50, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--model_name", default="resnet18", type=str)
    parser.add_argument("--num_workers", default=3, type=int)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--vocab_size", default=256, type=int)
    parser.add_argument("--num_embeddings", default=256, type=int)
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--feature_dim", default=64, type=int)
    parser.add_argument("--channels", default=3, type=int)
    parser.add_argument("--commitment_cost", default=0.6, type=int)

    args = parser.parse_args()

    train(args)