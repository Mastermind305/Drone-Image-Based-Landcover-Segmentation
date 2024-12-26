import os

import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset


class NepalDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.image_list = os.listdir(os.path.join(data_path, "images"))# "./output/512x512/"+images+
        # print("\n\n\n\n\n",self.image_list)
        self.mask_list = [image_name.replace(".jpg", "_mask.jpg") for image_name in self.image_list]
        # print("\n\n\n\n\n mask_list", self.mask_list, "\n\n\n\n")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_path, "images", self.image_list[idx])
        mask_name = os.path.join(
            self.data_path,
            "masks",
            self.image_list[idx].replace(".jpg", "_mask.jpg")
        )
        # print("adasdasssssssssssssssssssssssdasdasdassssssssssssssssssssssssssssssssdasdasdassssssssssssssssssssssdasd")
        print(img_name)
        # Load image and mask using tifffile
        img = tiff.imread(img_name)
        mask = tiff.imread(mask_name)

        # Convert image to RGB if it has more than 3 channels
        # if img.shape[-1] > 3:
        #     img = img[:, :, :3]

        # Convert image and mask to uint8 if needed
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask

    def get_mask_path(self, idx):
        mask_filename = self.mask_list[idx]
        mask_path = os.path.join(self.data_path, "masks", mask_filename)
        return mask_path


class NepalDataGenerator:
    def __init__(self, dataset, batch_size=16, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_idx = 0
        self.indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        num_batches, remainder = divmod(len(self.dataset), self.batch_size)
        return num_batches + int(remainder > 0)
        # return len(self.dataset) // self.batch_size

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self):
            raise StopIteration

        batch_indices = self.indices[self.current_idx * self.batch_size:(self.current_idx + 1) * self.batch_size]

        batch_images = []
        batch_masks = []

        for idx in batch_indices:
            image, mask = self.dataset[idx]
            batch_images.append(image)
            batch_masks.append(mask)

        self.current_idx += 1

        batch_images = torch.stack(batch_images)
        batch_masks = torch.stack(batch_masks)

        return batch_images, batch_masks
