import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon
from shapely.ops import unary_union
from torchvision.transforms import transforms
from tqdm import tqdm

from generator import NepalDataset, NepalDataGenerator


def poly_from_utm(polygon, transform):
    poly_pts = []

    poly = unary_union(polygon)
    for i in np.array(poly.exterior.coords):
        poly_pts.append(~transform * tuple(i))

    new_poly = Polygon(poly_pts)
    return new_poly


"""
def create_patches(image_path, mask_path, output_dir, patch_size, stride, boundary):
    # Open satellite image
    with rasterio.open(image_path, "r") as src:
        image = src.read()
        image_meta = src.meta
        image_crs = src.crs

    # Open shapefile mask
    mask_data = gpd.read_file(mask_path)
    mask_data = mask_data.to_crs(src.crs)

    # Get area of interest boundary to cut the patches
    min_lat, min_lon, max_lat, max_lon = boundary
    no_of_patches_saved = 0

    tqdm.write("Generating patches...")

    # Iterate over patches
    for i, (x, y) in tqdm(enumerate(generate_patch_coordinates(image.shape[1], image.shape[2], patch_size, stride))):
        # Calculate patch boundary
        patch_boundary = (
            min_lon + (max_lon - min_lon) * (x / image.shape[1]),
            min_lon + (max_lon - min_lon) * ((x + patch_size) / image.shape[1]),
            min_lat + (max_lat - min_lat) * (y / image.shape[2]),
            min_lat + (max_lat - min_lat) * ((y + patch_size) / image.shape[2])
        )

        # Create patch geometry
        patch_geometry = box(patch_boundary[0], patch_boundary[2], patch_boundary[1], patch_boundary[3])

        # Check if patch boundary intersects with any mask geometry
        intersecting_masks = mask_data[mask_data.intersects(patch_geometry)]

        # Print the number of intersecting masks for debugging
        tqdm.write(f"\nNumber of intersecting masks for patch {i}: {len(intersecting_masks)}")

        # Skip patch if no intersection with mask geometries
        if intersecting_masks.empty:
            continue

        # Get patch from image
        patch = image[:, x:x + patch_size, y:y + patch_size]

        # Create patch mask by rasterizing the intersecting mask geometries
        patch_mask = rasterize_masks(image_meta, intersecting_masks, patch_size, patch_size)

        # Save patch and patch mask
        patch_name = f"patch_{i}.tiff"
        patch_mask_name = f"patch_{i}_mask.tiff"
        patch_path = os.path.join(output_dir, "images", patch_name)
        patch_mask_path = os.path.join(output_dir, "masks", patch_mask_name)

        bin_image_meta = image_meta.copy()
        bin_image_meta.update({'count': patch.shape[0]})
        bin_image_meta.update({'height': patch.shape[1]})
        bin_image_meta.update({'width': patch.shape[2]})

        save_image_patch(patch, patch_path, bin_image_meta)
        tqdm.write(f"Image patch saved at: {output_dir}/images/{patch_name}")

        bin_mask_meta = image_meta.copy()
        bin_mask_meta.update({'count': 1})
        bin_mask_meta.update({'height': patch_mask.shape[0]})
        bin_mask_meta.update({'width': patch_mask.shape[1]})

        save_mask_patch(patch_mask, patch_mask_path, bin_mask_meta)

        tqdm.write(f"Mask patch saved at: {output_dir}/images/{patch_name}")
        no_of_patches_saved += 1
        # TODO: Remove this
        if (no_of_patches_saved == 10):
            break;
    tqdm.write(f"\nTotal number image and mask patches saved: {str(no_of_patches_saved)}")

"""


def create_patches(image_path, mask_path, output_dir, patch_size, stride):
    # Open satellite image
    with rasterio.open(image_path, "r") as src:
        image = src.read()
        image_meta = src.meta
        image_crs = src.crs

    # Open shapefile mask
    with rasterio.open(mask_path, "r") as src:
        mask_data = src.read()
        mask_meta = src.meta
        mask_crs = src.crs

    no_of_patches_saved = 0

    # tqdm.write("Generating patches...")

    # Iterate over patches
    for i, (x, y) in tqdm(enumerate(generate_patch_coordinates(image.shape[1], image.shape[2], patch_size, stride))):

        # Check if the patch size is smaller than the remaining image size
        if y + patch_size > image.shape[1] or x + patch_size > image.shape[2]:
            continue

        # if mask_patch.size == 0:
        #     tqdm.write(f"Skipping empty mask patch at ({x}, {y})")
        #     continue

        mask_patch = mask_data[:, y:y + patch_size, x: x + patch_size]
        tqdm.write(f"Processing patch {i} at coordinates ({x}, {y}) with size {mask_patch.shape}")
        tqdm.write(f"Mask patch shape: {mask_patch.shape}")

        # if mask_patch.size == 0:
        #     tqdm.write(f"Skipping empty mask patch at ({x}, {y})")
        #     continue

        # print(""mask_patch.shape)
        # one_hot_mask = np.zeros((2, patch_size, patch_size))
        # for class_value in [1, 2]:  # Assuming classes 1 and 2
        #     # class_indices = mask_patch == class_value
        #     class_indices = np.where(mask_patch == class_value, 1, 0)

        #     print(class_indices, class_value)
        #     # one_hot_mask[class_value - 1 , class_indices] = 1
        #     one_hot_mask[class_value - 1, :,:] = class_indices
        #     print(one_hot_mask)

                
        one_hot_mask = np.zeros((3, 513, 513))
        for class_value in [0, 1, 2]:  # Assuming classes 1 and 2
            # class_indices = mask_patch == class_value
            class_indices = np.where(mask_patch == class_value, class_value + 1, 0)
            # print("class_index",np.unique(mask))
            print(one_hot_mask.shape)
            print(class_indices, class_value)
            one_hot_mask[class_value, :,:] = class_indices

        print(one_hot_mask.shape)
        # Calculate the number of unique non-zero values in the patch
        num_intersecting_masks = len(np.unique(one_hot_mask.flatten())) - 1 if 0 in np.unique(one_hot_mask) else len( np.unique(one_hot_mask))
        tqdm.write(f"\nNumber of intersecting masks for patch {i}: {num_intersecting_masks}")


        # Skip patch if no intersection with mask geometries
        if np.count_nonzero(one_hot_mask) == 0:
            continue

        # Get patch from image
        image_patch = image[:, y:y + patch_size, x:x + patch_size]

        # Save patch and patch mask
        image_patch_name = f"Mission2patch_{i}.jpg"
        mask_patch_name = f"Mission2patch_{i}_mask.jpg"
        patch_path = os.path.join(output_dir, "images", image_patch_name)
        patch_mask_path = os.path.join(output_dir, "masks", mask_patch_name)

        bin_image_meta = image_meta.copy()
        bin_image_meta.update({'count': image_patch.shape[0]})
        bin_image_meta.update({'height': image_patch.shape[1]})
        bin_image_meta.update({'width': image_patch.shape[2]})

        save_image_patch(image_patch, patch_path, bin_image_meta)
        tqdm.write(f"Image patch saved at: {output_dir}/images/{image_patch_name}")

        bin_mask_meta = mask_meta.copy()
        bin_mask_meta.update({'count': one_hot_mask.shape[0]})
        bin_mask_meta.update({'height': one_hot_mask.shape[1]})
        print("\n\n\nheight\n\n\n",one_hot_mask.shape[0])
        bin_mask_meta.update({'width': one_hot_mask.shape[2]})

        save_mask_patch(one_hot_mask, patch_mask_path, bin_mask_meta)

        tqdm.write(f"Mask patch saved at: {output_dir}/images/{image_patch_name}")
        no_of_patches_saved += 1
    tqdm.write(f"\nTotal number image and mask patches saved: {str(no_of_patches_saved)}")


def visualize_image_mask(image_array, mask_array):
    # Check if the image is 5-channel
    if image_array.shape[-1] == 5:
        # Create a figure with 6 subplots (5 for image channels and 1 for the mask)
        fig, axes = plt.subplots(1, 6, figsize=(15, 5))
        fig.suptitle('Image and Mask Visualization')

        # Display each channel of the image in separate subplots
        for i in range(5):
            axes[i].imshow(image_array[:, :, i], cmap='gray')
            axes[i].set_title(f'Channel {i + 1}')
            axes[i].axis('off')
    else:
        # If the image is not 5-channel, display it as a single subplot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle('Image and Mask Visualization')

        axes[0].imshow(image_array, cmap='gray')
        axes[0].set_title('Image')
        axes[0].axis('off')

    # Display the mask in the last subplot
    axes[-1].imshow(mask_array, cmap='gray')
    axes[-1].set_title('Mask')
    axes[-1].axis('off')

    # Show the plots
    plt.show()


# def visualize(image, mask):

#     print(f"Image shape: {image.shape}")
#     print(f"Mask shape: {mask.shape}")
#     # Select bands 3, 2, and 1 for the RGB image
#     rgb_image = image[:, :, [1, 2, 0]]

#     # Plot the RGB image
#     fig, axs = plt.subplots(1, 2, figsize=(12, 6))

#     axs[0].imshow(image)
#     axs[0].set_title("Image")

#     # Plot the mask
#     axs[1].imshow(mask, cmap='gray')
#     axs[1].set_title("Mask")

#     # Show the plot
#     plt.show()
import numpy as np
import matplotlib.pyplot as plt

# def visualize(image, mask):
#     print(f"Image shape: {image.shape}")
#     print(f"Mask shape: {mask.shape}")
    
#     # Define custom colormap for visualization
#     colormap = {
#         0: (0, 0, 0),       # Background (black)
#         1: (255, 0, 0),      # Building (red)
#         2: (0, 128, 0),     # Woodland (green)
             
#     }
    
#     # Select bands 3, 2, and 1 for the RGB image
#     rgb_image = image[:, :, [1, 2, 0]]

#     # Plot the RGB image
#     fig, axs = plt.subplots(1, 2, figsize=(12, 6))

#     axs[0].imshow(rgb_image)
#     axs[0].set_title("Image")

#     # Convert the mask to RGB using the colormap
#     mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
#     for label, color in colormap.items():
#         mask_rgb[mask == label] = color

#     # Plot the mask
#     axs[1].imshow(mask_rgb)
#     axs[1].set_title("Mask")

#     # Show the plot
#     plt.show()



import numpy as np
import matplotlib.pyplot as plt

def visualize(image, mask):
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Define custom colormap for visualization
    colormap = {
        0: (0, 0, 0),       # Background (black)
        1: (255, 0, 0),     # Class 1 (red)
        2: (0, 128, 0),     # Class 2 (green)
    }
    
    # Select bands 3, 2, and 1 for the RGB image
    rgb_image = image[:, :, [1, 2, 0]]

    # Plot the RGB image
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    axs[0].imshow(rgb_image)
    axs[0].set_title("Image")

    # Convert the mask to RGB using the colormap
    # Initialize the RGB mask with background color
    mask_rgb = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)

    # Assign colors for class 1
    mask_rgb[mask[0] == 1] = colormap[1]
    # Assign colors for class 2
    mask_rgb[mask[1] == 2] = colormap[2]

    # Plot the mask
    axs[1].imshow(mask_rgb[:,:,0])
    axs[1].set_title("Mask")

    
    axs[2].imshow(mask_rgb[:,:,1])
    axs[2].set_title("Mask")
    # Show the plot
    plt.show()



def generate_patch_coordinates(image_width, image_height, patch_size, stride):
        for x in range(0, image_width - patch_size + 1, stride):
            for y in range(0, image_height - patch_size + 1, stride):
                yield x, y


""" 
def rasterize_masks(intersecting_masks, patch_geometry, image_width, image_height,transform):
    # Create mask image with same dimensions as the patch
    mask_image = np.zeros((1, image_width, image_height), dtype=np.uint8)

    # Iterate over intersecting mask geometries
    for idx, mask_geometry in intersecting_masks.iterrows():
        # Clip mask geometry with patch geometry
        clipped_geometry = mask_geometry.geometry.intersection(patch_geometry)

        # if clipped_geometry.is_empty:
        #     print(f"Empty geometry for index {idx}. Skipping...")
        #     continue

        # # Print clipped geometry and patch_geometry for debugging
        # print(f"Clipped Geometry for index {idx}:\n", clipped_geometry)
        # print(f"Patch Geometry for index {idx}:\n", patch_geometry)

        # Rasterize clipped geometry into the mask image
        mask_array = geometry_mask([clipped_geometry],
                                   out_shape=(image_width, image_height),
                                   transform=transform,
                                   invert=True)

        # Combine the mask_array with the mask_image using OR operation
        mask_image |= mask_array.astype(np.uint8)

        # Print the mask_array for debugging
        print(f"Mask Array for index {idx}:\n", mask_array)

    return mask_image

 """

"""
def rasterize_masks(intersecting_masks, patch_geometry, image_width, image_height, transform, num_classes=22):
    # Create an empty numpy array to store the mask
    mask = np.zeros((num_classes, image_height, image_width), dtype=np.uint8)

    # Iterate over each class in the intersecting masks
    for i, class_name in enumerate(intersecting_masks['Class'].unique()):
        # Get the subset of data for the current class
        class_data = intersecting_masks[intersecting_masks['Class'] == class_name]

        # Convert the class data to a list of geometries
        class_geoms = list(class_data['geometry'])

        # Rasterize the class geometries to a binary mask
        class_mask = geometry_mask(
            class_geoms,
            out_shape=(image_height, image_width),
            transform=transform,
            invert=True
        )

        # Store the class mask in the output mask array
        mask[i] = class_mask.astype(np.uint8)

    return mask
"""


def rasterize_masks(image_meta, intersecting_masks, image_height, image_width):
    poly_shp = []
    im_size = (image_height, image_width)
    for num, row in intersecting_masks.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            poly = poly_from_utm(row['geometry'], image_meta['transform'])
            poly_shp.append(poly)
        else:
            for p in row['geometry']:
                poly = poly_from_utm(p, image_meta['transform'])
                poly_shp.append(poly)

    mask = rasterize(shapes=poly_shp,
                     out_shape=im_size, transform=image_meta['transform'], )
    mask = mask.astype("uint16")

    return mask


# def rasterize_masks(image_meta, intersecting_masks, image_height, image_width):
#     mask = np.zeros((1, image_height, image_width), dtype=np.uint8)
#
#     for idx, mask_geometry in intersecting_masks.iterrows():
#         mask_array = rasterize([mask_geometry.geometry],
#                                out_shape=(image_height, image_width))
#
#         mask |= mask_array.astype(np.uint8)
#
#     return mask


def save_image_patch(patch, patch_path, image_meta):
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(patch_path), exist_ok=True)

    # Save patch as TIFF image
    with rasterio.open(patch_path, 'w', **image_meta) as dst:
        dst.write(patch)


def save_mask_patch(patch, patch_path, mask_meta):
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(patch_path), exist_ok=True)
    #  (after coorection)


    # Save patch as TIFF image
    with rasterio.open(patch_path, 'w', **mask_meta) as dst:
        dst.write(patch)


def prepare_dataset():
    image_path = './datasets/planet_dataset/composite.tif'
    mask_path = './datasets/custom_dataset/shape-files/Landcover_Clip.shp'
    output_dir = "datasets/custom_dataset/output/512x512"
    patch_size = 512
    stride = 256
    boundary = [3050511, 354090, 3066774, 375485]
    create_patches(image_path, mask_path, output_dir, patch_size, stride)


def test_data_generator():
    # Create the dataset
    data_path = "./datasets/custom_dataset/output/512x512/"
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5], std=[0.5]),
    #     transforms.Lambda(lambda x: x * 255),
    # ])

    dataset = NepalDataset(data_path, transform=transform)

    # Create the data generator
    batch_size = 16
    shuffle = True
    data_generator = NepalDataGenerator(dataset, batch_size=batch_size, shuffle=shuffle)
    images, masks = data_generator.__next__()  # Get the batch of images and masks

    for i in range(0, 3):
        image = images[i].permute(1, 2, 0).numpy()  # Access individual image and convert to numpy array
        mask = masks[i].squeeze().numpy()  # Access individual mask and convert to numpy array
        visualize(image, mask)


   
