import os
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset


class NepalDataset(Dataset):
    def __init__(self, data_path, transform=None, training=True):
        self.data_path = data_path
        self.transform = transform
        self.training = training
        self.image_list = self._getimages()
        if self.training:
            self.images = self.image_list[1]
        else:
            self.images = self.image_list[2]  # Assuming validation images are in the 3rd list
        self.mask_list = [image_name.replace(".jpg", "_mask.jpg") for image_name in self.images]
        self.last_successful_data = None

    def _getimages(self):
        return [os.listdir(os.path.join(self.data_path, "images", i)) for i in os.listdir(os.path.join(self.data_path, "images"))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = [os.path.join(self.data_path, "images", i, self.images[idx]) for i in os.listdir(os.path.join(self.data_path, "images")) if os.path.exists(os.path.join(self.data_path, "images", i, self.images[idx]))]
        mask_name = [os.path.join(self.data_path, "masks", i, self.images[idx].replace(".jpg", "_mask.jpg")) for i in os.listdir(os.path.join(self.data_path, "masks")) if os.path.exists(os.path.join(self.data_path, "masks", i, self.images[idx].replace(".jpg", "_mask.jpg")))]
        
        try:
            # # Debug: Print the image and mask paths
            # print(f"Loading image: {img_name}")
            # print(f"Loading mask: {mask_name}")

            img = tiff.imread(img_name[0])
            mask = tiff.imread(mask_name[0])

            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)

            if self.transform is not None:
                img = self.transform(img)
                mask = self.transform(mask)

            # Debug: Print the shape of the loaded images and masks
            # print(f"Image shape: {img.shape}")
            # print(f"Mask shape: {mask.shape}")

            self.last_successful_data = (img[:3, :, :], mask[:,:,:])
            return img[:3, :, :], mask[:,:,:]
        except Exception as e:
            # print("\n\n\nWarning encountered", e)
            # print(f"Image name: {img_name}")
            # print(f"Mask name: {mask_name}")
            # print(f"Last successful data: {self.last_successful_data}")
            return self.last_successful_data

    def get_mask_path(self, idx):
        mask_filename = self.mask_list[idx]
        mask_path = os.path.join(self.data_path, "masks", mask_filename)
        return mask_path

class NepalDataGenerator:
    def __init__(self, dataset, batch_size=1, shuffle=True):
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
            if self.dataset[idx] is None:
                continue
            image, mask = self.dataset[idx]
            batch_images.append(image)
            batch_masks.append(mask)

        self.current_idx += 1

        if len(batch_images) == 0:  # Skip empty batches
            return self.__next__()

        batch_images = torch.stack(batch_images)
        batch_masks = torch.stack(batch_masks)

        # Debug: Print batch shapes
        # print(f"Batch images shape: {batch_images.shape}")
        # print(f"Batch masks shape: {batch_masks.shape}")

        return batch_images, batch_masks

