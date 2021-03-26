import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from vae import VQVae, OpenAIDiscreteVAE


def build_tokens(args):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_transforms = transforms.Compose(
        [transforms.CenterCrop(32),
         transforms.ToTensor(), normalize])

    train_dataset = datasets.CIFAR10(args.data_dir,
                                     train=True,
                                     transform=data_transforms,
                                     download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size)

    test_dataset = datasets.CIFAR10(args.data_dir,
                                    train=False,
                                    transform=data_transforms,
                                    download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size)

    if args.architecture == "dall-e":
        model = OpenAIDiscreteVAE()
    else:
        model = VQVae(args.vocab_size, args.embedding_dim, args.num_blocks,
                      args.feature_dim, args.channels)

    model.to(device)

    train_out_dir = os.path.join(args.data_dir, "ViTBert-Tokens", "train")
    if not os.path.isdir(train_out_dir):
        os.makedirs(train_out_dir)

    example_counter = 0
    for (images, labels) in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Retrive  quanitzed targets from encoder
        encodings = model.get_codebook_indices(images)

        # Append cls token to start of encodings
        encodings += 1
        cls_tokens = torch.zeros(len(images), 1, device=device)
        cls_encodings = torch.cat([cls_tokens, encodings], dim=1).long()

        # Save tokens to file
        file_names = [
            os.path.join(train_out_dir, f"train_{i}.pt")
            for i in range(example_counter, example_counter + len(images))
        ]
        for file_name, encoding, label in zip(file_names, cls_encodings,
                                              labels):
            datum = {"tokens": encoding, "label": label}
            torch.save(datum, file_name)

        example_counter += len(images)

    test_out_dir = os.path.join(args.data_dir, "ViTBert-Tokens", "test")
    if not os.path.isdir(test_out_dir):
        os.makedirs(test_out_dir)

    example_counter = 0
    for (images, labels) in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Retrive  quanitzed targets from encoder
        encodings = model.get_codebook_indices(images)

        # Append cls token to start of encodings
        encodings += 1
        cls_tokens = torch.ones(len(images), 1, device=device)
        cls_encodings = torch.cat([cls_tokens, encodings], dim=1).long()

        # Save tokens to file
        file_names = [
            os.path.join(test_out_dir, f"test_{i}.pt")
            for i in range(example_counter, example_counter + len(images))
        ]
        for file_name, encoding, label in zip(file_names, cls_encodings,
                                              labels):
            datum = {"tokens": encoding, "label": label}
            torch.save(datum, file_name)

        example_counter += len(images)
