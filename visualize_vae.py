import argparse
from vae import visualize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize VQ-VAE')

    # Directory args
    parser.add_argument("--data_dir",
                        default="./data",
                        help="location dataset is stored",
                        type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)

    # Plot args
    parser.add_argument("--random_state", type=int)
    parser.add_argument("--num_examples", default=16, type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--dataset", default="image-net-tar", type=str)

    # Vae args
    parser.add_argument("--num_codebook_indeces", default=1000, type=int)
    parser.add_argument("--embedding_dim", default=512, type=int)
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--feature_dim", default=128, type=int)
    parser.add_argument("--channels", default=3, type=int)
    parser.add_argument("--commitment_cost", default=0.5, type=float)

    args = parser.parse_args()

    visualize(args)