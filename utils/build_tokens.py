import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from vae import vae


def build_tokens(args):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose(
        [transforms.CenterCrop(32),
         transforms.ToTensor(), normalize])

    train_dataset = datasets.CIFAR10(args.data_dir,
                                     train=True,
                                     transform=train_transforms,
                                     download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size)

    model = vae.VQVae(args.vocab_size, args.num_embeddings, args.num_blocks,
                      args.feature_dim, args.channels)
    model.to(device)

    for (images, _) in tqdm(train_loader):
        encodings = model.get_encodings(images)
        print(encodings)
