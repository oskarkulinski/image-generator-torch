from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image

import parameters as params


class ImageGeneratorDataset(Dataset):
    """
    Custom data loader for loading images from multiple directories with caching support.
    Images are normalized to [-1, 1] range and cached for faster subsequent loading.
    """
    def __init__(self, root_dir, transform="default"):
        """
       Args:
           root_dir (str): Directory with all the image folders
           transform (callable, optional): Additional transformations
       """
        self.root_dir = Path(root_dir)
        if transform == "default":
            self.transform = transforms.Compose([
                transforms.Resize((params.image_height, params.image_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],  # Convert to [-1, 1]
                                     std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = None

        self.image_files = []

        for i, dir_path in enumerate(sorted(self.root_dir.glob('*/'))):
            image_paths = list(dir_path.glob('*.jpg')) + list(dir_path.glob('*.png'))
            self.image_files.extend(image_paths)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

def load_image_dataset(root_dir, batch_size=params.batch_size, num_workers=4):
    """
    Create data loaders for training and validation

    Args:
        root_dir (str): Path to the root directory containing image subdirectories
        batch_size (int): Batch size for the data loader
        num_workers (int): Number of workers for parallel data loading

    Returns:
        DataLoader: PyTorch DataLoader object
    """
    # Create dataset
    dataset = ImageGeneratorDataset(
        root_dir=root_dir,
        transform="default"
    )


    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return data_loader