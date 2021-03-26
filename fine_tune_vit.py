import argparse
from vit import fine_tune

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VQ-VAE')

    # Directory args
    parser.add_argument("--data_dir",
                        default="./data/ViTBert-Tokens",
                        help="location dataset is stored",
                        type=str)
    parser.add_argument("--mpp_ckpt", required=True, type=str)
    parser.add_argument("--checkpoint_dir",
                        default="./linear_head_ckpt",
                        type=str)
    parser.add_argument("--summary_dir",
                        default="./linear_head_runs",
                        type=str)

    # Training args
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--val_size", default=0.15, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--step_size", default=30, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--print_freq", default=50, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--num_workers", default=3, type=int)

    # Vae args
    parser.add_argument("--architecture", type=str)
    parser.add_argument("--embedding_dim", default=256, type=int)
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--feature_dim", default=64, type=int)
    parser.add_argument("--channels", default=3, type=int)
    parser.add_argument("--commitment_cost", default=0.6, type=int)
    parser.add_argument("--num_codebook_indeces", default=1024, type=int)

    # Vit args
    parser.add_argument("--vocab_size", default=256, type=int)
    parser.add_argument("--extension", default="pt", type=str)
    parser.add_argument("--dim", default=1024, type=int)
    parser.add_argument("--depth", default=8, type=int)
    parser.add_argument("--heads", default=8, type=int)
    parser.add_argument("--mlp_dim", default=2048, type=int)
    parser.add_argument("--dim_head", default=64, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--emb_dropout", default=0.0, type=float)
    parser.add_argument("--pad_token_id", default=0, type=int)
    parser.add_argument("--cls_token_id", default=1, type=int)

    # Mpp args
    parser.add_argument("--mask_prob", default=0.15, type=float)
    parser.add_argument("--replace_prob", default=0.9, type=float)
    parser.add_argument("--random_token_prob", default=0.0, type=float)
    parser.add_argument("--mask_token_id", default=2, type=int)
    parser.add_argument("--mask_ignore_token_ids", default=[], nargs="*")

    # Classifier args
    parser.add_argument("--out_dim", default=10, type=int)
    parser.add_argument("--freeze_transformer",
                        default=True,
                        type=lambda x: (str(x).lower() != 'false'))

    args = parser.parse_args()

    fine_tune(args)