import torch
import os


class TokensDataset(torch.utils.data.Dataset):
    """VQ-VAE tokens dataset"""
    def __init__(self, root_dir, extension="pt"):
        if not os.path.isdir(root_dir):
            raise ValueError("Root dir specified does not exist")

        self.root_dir = root_dir
        self.extension = extension
        self.samples = self._get_files()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_path = os.path.join(self.root_dir, self.samples[idx])
        datum = torch.load(data_path)
        tokens = datum["tokens"]
        label = datum["label"]
        return tokens, label

    def _get_files(self):
        files = os.listdir(self.root_dir)
        files = [
            file for file in files if file.lower().endswith(self.extension)
        ]
        return files
