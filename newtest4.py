import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from deemodel import prepare_model

def infer_single_image(model, image_path, transform=None, threshold=0.5,alpha=0.4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Load and transform the image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Convert grayscale to RGB
    original_image = image.copy()  # Keep a copy of the original image for overlay
    if transform is not None:
        image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():  # Disable gradient computation
        output = model(image)["out"]
        
        # Check for expected output shape (1, 3, H, W)
        if output.shape[1] != 3:
            raise ValueError("Unexpected output shape. Expected (1, 3, H, W)")

        output = torch.sigmoid(output)  # Apply sigmoid to get probabilities
        output = (output > threshold).float()  # Convert probabilities to binary predictions

    # Remove batch dimension and move to CPU
    output = output.squeeze().cpu().numpy()

    # Create overlay images for each channel
    red_overlay = np.zeros((output.shape[1], output.shape[2], 3), dtype=np.uint8)     # Building class
    green_overlay = np.zeros((output.shape[1], output.shape[2], 3), dtype=np.uint8)   # Woodland class
    blue_overlay = np.zeros((output.shape[1], output.shape[2], 3), dtype=np.uint8)    # Unlabeled class

    red_overlay[output[0] == 1] = [255, 0, 0]  # Red for the first channel
    green_overlay[output[1] == 1] = [0, 255, 0]  # Green for the second channel
    blue_overlay[output[2] == 1] = [0, 0, 255]  # Blue for the third channel

    # Convert original image to numpy array
    original_image_np = np.array(original_image)

    # # Combine the overlays with the original image
    # combined_overlay = original_image_np.copy()
    # combined_overlay[output[0] == 1] = combined_overlay[output[0] == 1] * 0.5 + red_overlay[output[0] == 1] * 0.5
    # combined_overlay[output[1] == 1] = combined_overlay[output[1] == 1] * 0.5 + green_overlay[output[1] == 1] * 0.5
    # combined_overlay[output[2] == 1] = combined_overlay[output[2] == 1] * 0.5 + blue_overlay[output[2] == 1] * 0.5

    combined_overlay = original_image_np.copy()
    combined_overlay[output[0] == 1] = combined_overlay[output[0] == 1] * (1 - alpha) + red_overlay[output[0] == 1] * alpha
    combined_overlay[output[1] == 1] = combined_overlay[output[1] == 1] * (1 - alpha) + green_overlay[output[1] == 1] * alpha
    combined_overlay[output[2] == 1] = combined_overlay[output[2] == 1] * (1 - alpha) + blue_overlay[output[2] == 1] * alpha


    # Plot the image and the predicted mask
    plt.figure(figsize=(12, 6))
    
    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")

    # Display the combined overlay image
    plt.subplot(1, 2, 2)
    plt.imshow(combined_overlay)
    plt.title("Predicted Mask Overlay")

    plt.show()

    return output

def test_model(model, image_path, transform=None):
    # Load the saved best model weights
    best_model_weights = torch.load('bestmodeloverall22.pth')
    # Create a new model instance
    model.load_state_dict(best_model_weights)

    # Perform inference on each test image
    infer_single_image(model, image_path, transform)

# Example usage:
def infer_single_image_with_deep():
    # Create PAN model
    deep_model = prepare_model()

    # Path to the image
    image_path = r"C:\Users\bhatt\OneDrive\Desktop\semprj\aerial-satellite-imagery-segmentation-nepal\Output_mg\513x513\images\test\Mission4patch_509.jpg"
    
    # Transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Infer the model on a single image
    test_model(deep_model, image_path, transform)

# Call the inference function for single image
infer_single_image_with_deep()
