import rasterio
import tifffile
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

def getimages (data_path):
    return [os.path.join(data_path, i) for i in os.listdir(os.path.join(data_path))]

def getimglist(data_path):
    return [i for i in os.listdir(os.path.join(data_path))]


mask_path =  r"C:\Users\bhatt\OneDrive\Desktop\semprj\aerial-satellite-imagery-segmentation-nepal\output12\513x513\masks"
temp_mask_path = r"C:\Users\bhatt\OneDrive\Desktop\semprj\aerial-satellite-imagery-segmentation-nepal\output12\513x513\images"

# img_path = getimages(temp_mask_path)
img_list = getimglist(temp_mask_path)
print(img_list)
for i in img_list:
    img_path = os.path.join(temp_mask_path, i)
    with rasterio.open(img_path) as src:
        mask = src.read()
        # mask_transform = src.transform
    zero_count = np.count_nonzero(mask == 0)
    total_count = mask.size
    zero_percentage = (zero_count / total_count) * 100

    # print("zero count",zero_count)
    # print(stats.mode(mask.flatten()).mode[0])

    print(stats.mode(mask.flatten()), zero_percentage)
    if stats.mode(mask.flatten()) == 0 or zero_percentage>50:
        os.remove(img_path)
        os.remove(os.path.join(mask_path, i.replace(".jpg", "_mask.jpg")))
        print("Deleted", img_path, "and its mask")
