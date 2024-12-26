import torch.nn as nn

from torchvision.models.segmentation import deeplabv3_resnet50
 
def prepare_model(num_classes=2):
    model = deeplabv3_resnet50(weights='DEFAULT')
    model.classifier[-1] = nn.Conv2d(256, num_classes, 1)
    model.aux_classifier[-1] = nn.Conv2d(256, num_classes, 1)
    # model.train()
    return model

# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision.models.segmentation import deeplabv3_resnet50
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image

# class PrepareModel(nn.Module):
#     def __init__(self, num_classes=2):
#         super(PrepareModel, self).__init__()
#         self.num_classes = num_classes
#         self.model = deeplabv3_resnet50(weights='DEFAULT')
        
#         # Modify the classifier and auxiliary classifier
#         self.model.classifier[-1] = nn.Conv2d(256, self.num_classes, 1)
#         if self.model.aux_classifier:
#             self.model.aux_classifier[-1] = nn.Conv2d(256, self.num_classes, 1)

#     def forward(self, x):
#         return self.model(x)

# def visualize_model(model, image):
#     model.eval()
#     with torch.no_grad():
#         output = model(image.unsqueeze(0))  # Add batch dimension

#     # Get the output prediction
#     output_predictions = output['out'].argmax(1).squeeze().cpu().numpy()

#     # Plot the input image and the output prediction
#     fig, ax = plt.subplots(1, 2, figsize=(12, 6))

#     # Plot input image
#     ax[0].imshow(image.permute(1, 2, 0).cpu().numpy())
#     ax[0].set_title("Input Image")
#     ax[0].axis('off')

#     # Plot output prediction
#     ax[1].imshow(output_predictions, cmap='jet', alpha=0.7)
#     ax[1].set_title("Output Prediction")
#     ax[1].axis('off')

#     plt.show()

# # Example usage
# if __name__ == "__main__":
#     # Initialize the model
#     model = PrepareModel(num_classes=2)
    
#     # Load an example image and preprocess it
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
    
#     # Replace 'path_to_image' with the actual path to your image
#     image = Image.open(r'output\512x512\images\train\patch_45.jpg').convert('RGB')
#     image = transform(image)
    
#     # Visualize the model output
#     visualize_model(model, image)