import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from text_preprocess import text_to_sequence, build_vocab, vocab

# Absolute paths for annotations
data_dir = 'data'
train_dir = os.path.join(data_dir, 'train2017')
val_dir = os.path.join(data_dir, 'val2017')
train_annotations_path = 'data/annotations/captions_train2017.json'
val_annotations_path = 'data/annotations/captions_val2017.json'

# Print paths to verify
print(f"Train annotations path: {train_annotations_path}")
print(f"Val annotations path: {val_annotations_path}")

# Verify if the paths exist
print(f"Train annotations file exists: {os.path.isfile(train_annotations_path)}")
print(f"Val annotations file exists: {os.path.isfile(val_annotations_path)}")

# Open the annotations files
with open(train_annotations_path, 'r') as f:
    train_annotations = json.load(f)

with open(val_annotations_path, 'r') as f:
    val_annotations = json.load(f)

# Build the vocabulary from the training captions
train_captions = [ann['caption'] for ann in train_annotations['annotations']]
build_vocab(train_captions)
print(f"Vocabulary size: {len(vocab)}")
print(f"Sample vocabulary items: {list(vocab.items())[:10]}")

# Define the transformation for the images
transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomAffine(degrees=10, scale=(0.9, 1.1), translate=(0.1, 0.1)),
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    T.RandomHorizontalFlip(0.5),
    T.RandomCrop(224, padding_mode="reflect", pad_if_needed=True),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform(image)

def preprocess_caption(caption):
    return text_to_sequence(caption)

class COCO_Dataset(Dataset):
    def __init__(self, annotations, image_dir, transforms=None):
        self.annotations = annotations['annotations']
        self.image_dir = image_dir
        self.transform = transforms
        self.image_info = {image['id']: image['file_name'] for image in annotations['images']}
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        annotation = self.annotations[index]
        img_id = annotation['image_id']
        caption = annotation['caption']
        img_name = self.image_info[img_id]
        img_path = os.path.join(self.image_dir, img_name)
        show_sample_image(img_path)  # Show sample image before transformation
        image = preprocess_image(img_path)
        caption_sequence = preprocess_caption(caption)
        return image, torch.tensor(caption_sequence)

def collate_fn(batch):
    images, captions = zip(*batch)
    
    # Pad captions to the maximum length in the batch
    max_len = max(len(caption) for caption in captions)
    padded_captions = torch.zeros(len(captions), max_len, dtype=torch.long)
    
    for i, caption in enumerate(captions):
        end = len(caption)
        padded_captions[i, :end] = caption
    
    images = torch.stack(images, dim=0)
    return images, padded_captions

def show_sample_image(image_path):
    image = Image.open(image_path).convert('RGB')
    plt.imshow(image)
    plt.title("Sample Image Before Transformation")
    plt.axis('off')
    plt.show()

def show_images_and_captions(dataset, num_images=5):
    fig, axs = plt.subplots(num_images, 1, figsize=(10, 20))
    for i in range(num_images):
        image, caption = dataset[i]
        image = image.permute(1, 2, 0).numpy()
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)

        caption_tokens = [list(vocab.keys())[list(vocab.values()).index(token.item())] for token in caption if token.item() in list(vocab.values())]

        axs[i].imshow(image)
        axs[i].set_title(f"Caption: {' '.join(caption_tokens)}")
        axs[i].axis('off')
        print(f"Image {i+1} caption tokens: {caption_tokens}")

    plt.show()

# Example usage
train_dataset = COCO_Dataset(train_annotations, train_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

show_images_and_captions(train_dataset, num_images=5)

for images, captions in train_loader:
    print(images.shape, captions.shape)
    break
