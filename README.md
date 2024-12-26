Drone Image-Based Landcover Segmentation

Overview

This project focuses on segmenting landcover using drone imagery of Butwal Ward No. 6. The goal is to process high-resolution images, train a segmentation model, and evaluate its performance in identifying different landcover classes.

Project Structure

Data: The data consists of high-resolution .tif images and associated shapefiles.

Model: DeepLab v3 is used as the segmentation model, implemented in deemodel.py.

Training: The training pipeline is managed using the script train4.py.

Preprocessing: Preprocessing steps, including rasterizing shapefiles and patch extraction, are handled in the Jupyter Notebook project.ipynb.

Testing: Model testing and evaluation are performed using newtest4.py.

Getting Started

Prerequisites

Ensure you have the following installed:

Python 3.8+

Required Python libraries (listed in requirements.txt)

Install the dependencies:

pip install -r requirements.txt

Dataset

The dataset includes:

.tif files for drone imagery.

Shapefiles (.shp) containing ground truth data for segmentation.

Place the data in the data/ directory. Ensure the directory structure is as follows:

data/
├── imagery/
│   ├── image1.tif
│   ├── image2.tif
├── shapefiles/
│   ├── ground_truth1.shp
│   ├── ground_truth2.shp

Workflow

Preprocessing

The preprocessing steps include:

Rasterizing Shapefiles: Convert the shapefiles into raster format aligned with the .tif images.

Patch Extraction: Cut the .tif images into smaller patches suitable for training.

Run the preprocessing steps in project.ipynb.

Training

To train the DeepLab v3 model:

Open and configure train4.py with the desired parameters (e.g., epochs, learning rate).

Run the script:

python train4.py

Testing

To evaluate the trained model:

Ensure the model weights are saved in the appropriate location.

Run the testing script:

python newtest4.py

File Details

Scripts

deemodel.py: Contains the implementation of the DeepLab v3 model.

train4.py: Script for training the segmentation model.

newtest4.py: Script for testing the model on new data.

Notebooks

project.ipynb: Includes preprocessing steps such as rasterization and patch generation.

Data

.tif files: High-resolution drone images.

.shp files: Ground truth annotations for landcover.

Results

The trained DeepLab v3 model achieves segmentation of landcover classes with high accuracy. Visualizations and metrics are stored in the results/ directory.

Future Work

Experiment with other segmentation models.

Use data augmentation techniques to improve model generalization.

Expand the dataset to include other wards or regions.

License

This project is licensed under the MIT License. See LICENSE for details.

Contact

For questions or feedback, please contact [your-email@example.com].



