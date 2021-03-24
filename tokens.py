import argparse
from utils import build_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Building VQ-VAE tokens for ViTBert')

    parser.add_argument("--data_dir",
                        default="./data",
                        help="location dataset is stored",
                        type=str)
    parser.add_argument("--architecture", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--model_name", default="resnet18", type=str)
    parser.add_argument("--num_workers", default=3, type=int)
    parser.add_argument("--vocab_size", default=256, type=int)
    parser.add_argument("--num_embeddings", default=256, type=int)
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--feature_dim", default=64, type=int)
    parser.add_argument("--channels", default=3, type=int)

    args = parser.parse_args()

    build_tokens(args)