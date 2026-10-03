import os
import tarfile
import urllib.request
import ssl
from pathlib import Path
from PIL import Image

# Workaround for macOS SSL certificate verification errors
ssl._create_default_https_context = ssl._create_unverified_context
import torch
from torch.utils.data import Dataset
from torchvision import transforms

DATASET_URL = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"

class FGVCAircraft(Dataset):
    def __init__(self, root_dir, split='train', transform=None, download=True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        self.dataset_dir = self.root_dir / "fgvc-aircraft-2013b"
        self.image_dir = self.dataset_dir / "data" / "images"
        
        if download:
            self._download_and_extract()
            
        if not self.dataset_dir.exists():
            raise RuntimeError(f"Dataset not found at {self.dataset_dir}. Use download=True to download it.")
            
        self.samples = []
        self.classes = {'variant': [], 'family': [], 'maker': []}
        self.class_to_idx = {'variant': {}, 'family': {}, 'maker': {}}
        
        self._load_data()
        
    def _download_and_extract(self):
        if self.dataset_dir.exists():
            return
            
        self.root_dir.mkdir(parents=True, exist_ok=True)
        tar_path = self.root_dir / "fgvc-aircraft-2013b.tar.gz"
        
        if not tar_path.exists():
            print(f"Downloading FGVC-Aircraft dataset to {tar_path}...")
            urllib.request.urlretrieve(DATASET_URL, tar_path)
            print("Download complete.")
            
        print("Extracting dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=self.root_dir)
        print("Extraction complete.")

    def _load_data(self):
        # We need to map image IDs to their labels
        # Depending on the split, we use train, val, or test txt files
        
        split_file_suffix = f"{self.split}.txt"
        
        def load_labels(file_level, dict_key):
            labels_dict = {}
            filepath = self.dataset_dir / "data" / f"images_{file_level}_{split_file_suffix}"
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        img_id, label = parts
                        labels_dict[img_id] = label
                        if label not in self.class_to_idx[dict_key]:
                            self.class_to_idx[dict_key][label] = len(self.class_to_idx[dict_key])
                            self.classes[dict_key].append(label)
            return labels_dict

        variant_labels = load_labels('variant', 'variant')
        family_labels = load_labels('family', 'family')
        maker_labels = load_labels('manufacturer', 'maker')
        
        # Load bounding boxes
        boxes_dict = {}
        with open(self.dataset_dir / "data" / "images_box.txt", 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    img_id = parts[0]
                    # xmin, ymin, xmax, ymax
                    boxes_dict[img_id] = [int(p) for p in parts[1:]]

        # Combine into samples
        for img_id in variant_labels.keys():
            if img_id in family_labels and img_id in maker_labels:
                self.samples.append({
                    'img_id': img_id,
                    'path': self.image_dir / f"{img_id}.jpg",
                    'variant_idx': self.class_to_idx['variant'][variant_labels[img_id]],
                    'family_idx': self.class_to_idx['family'][family_labels[img_id]],
                    'maker_idx': self.class_to_idx['maker'][maker_labels[img_id]],
                    'box': boxes_dict.get(img_id, None)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'variant': sample['variant_idx'],
            'family': sample['family_idx'],
            'maker': sample['maker_idx'],
            'box': torch.tensor(sample['box']) if sample['box'] else torch.tensor([-1, -1, -1, -1])
        }
        
    def get_num_classes(self, level):
        return len(self.classes[level])

if __name__ == "__main__":
    # Test script to download and verify
    print("Testing Dataset Loader...")
    dataset = FGVCAircraft(root_dir="./data", split='train', download=True)
    print(f"Loaded {len(dataset)} training samples.")
    print(f"Number of Variants: {dataset.get_num_classes('variant')}")
    print(f"Number of Families: {dataset.get_num_classes('family')}")
    print(f"Number of Makers: {dataset.get_num_classes('maker')}")
    
    sample = dataset[0]
    print(f"First sample shapes - Image: {sample['image'].size}, Variant: {sample['variant']}, Family: {sample['family']}, Maker: {sample['maker']}")
