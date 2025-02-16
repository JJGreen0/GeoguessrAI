#!/usr/bin/env python
import cv2
import numpy as np
import pyautogui
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import pandas as pd

# Load the CSV file
df = pd.read_csv("metrics.csv")

# Extract the class names from the 'Class' column
CLASSES = ['Aland',
 'Albania',
 'American Samoa',
 'Andorra',
 'Antarctica',
 'Argentina',
 'Armenia',
 'Australia',
 'Austria',
 'Bangladesh',
 'Belarus',
 'Belgium',
 'Bermuda',
 'Bhutan',
 'Bolivia',
 'Botswana',
 'Brazil',
 'Bulgaria',
 'Cambodia',
 'Canada',
 'Chile',
 'China',
 'Colombia',
 'Costa Rica',
 'Croatia',
 'Curacao',
 'Czechia',
 'Denmark',
 'Dominican Republic',
 'Ecuador',
 'Egypt',
 'Estonia',
 'Eswatini',
 'Faroe Islands',
 'Finland',
 'France',
 'Germany',
 'Ghana',
 'Gibraltar',
 'Greece',
 'Greenland',
 'Guam',
 'Guatemala',
 'Hong Kong',
 'Hungary',
 'Iceland',
 'India',
 'Indonesia',
 'Iraq',
 'Ireland',
 'Isle of Man',
 'Israel',
 'Italy',
 'Japan',
 'Jersey',
 'Jordan',
 'Kenya',
 'Kyrgyzstan',
 'Laos',
 'Latvia',
 'Lebanon',
 'Lesotho',
 'Lithuania',
 'Luxembourg',
 'Macao',
 'Madagascar',
 'Malaysia',
 'Malta',
 'Martinique',
 'Mexico',
 'Monaco',
 'Mongolia',
 'Montenegro',
 'Mozambique',
 'Myanmar',
 'Nepal',
 'Netherlands',
 'New Zealand',
 'Nigeria',
 'North Macedonia',
 'Northern Mariana Islands',
 'Norway',
 'Pakistan',
 'Palestine',
 'Paraguay',
 'Peru',
 'Philippines',
 'Pitcairn Islands',
 'Poland',
 'Portugal',
 'Puerto Rico',
 'Qatar',
 'Reunion',
 'Romania',
 'Russia',
 'San Marino',
 'Senegal',
 'Serbia',
 'Singapore',
 'Slovakia',
 'Slovenia',
 'South Africa',
 'South Georgia and South Sandwich Islands',
 'South Korea',
 'South Sudan',
 'Spain',
 'Sri Lanka',
 'Svalbard and Jan Mayen',
 'Sweden',
 'Switzerland',
 'Taiwan',
 'Tanzania',
 'Thailand',
 'Tunisia',
 'Turkey',
 'US Virgin Islands',
 'Uganda',
 'Ukraine',
 'United Arab Emirates',
 'United Kingdom',
 'United States',
 'Uruguay',
 'Venezuela',
 'Vietnam']

# Define your class names for location prediction.
# For example, if your model predicts 5 countries, list them here.

def load_model(model_path):
    """
    Reinitialize the model architecture (MobileNetV2 with a custom classifier)
    and load the trained weights from a .pth file.
    """
    # Use a lightweight pre-trained model (MobileNetV2) for speed.
    model = models.mobilenet_v2(pretrained=True)

    # Freeze the feature extractor layers to speed up training.
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier to match the number of classes from the CSV.
    in_features = model.classifier[1].in_features
    num_classes = 124  # Should be 124 if CLASSES contains 124 names.
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Load the trained weights.
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Set to evaluation mode
    return model

def main():
    # Path to the model's weights (adjust if necessary)
    model_path = "model_epoch_3.pth"
    model = load_model(model_path)

    # Define the image transformation (must match training/inference)
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Use the same resolution as during training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Define the screen region to capture (left, top, width, height)
    # Adjust these coordinates to match the area where Google Maps appears on your screen.
    region = (100, 100, 1800, 1600)
    
    print("Starting real-time inference. Press 'q' to quit.")
    while True:
         # Capture the specified region for inference.
        screenshot = pyautogui.screenshot(region=region)
        img_np = np.array(screenshot)
        
        # For inference display, convert the region image from RGB to BGR.
        display_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Also capture the full screen to visualize the capture area.
        screenshot_full = pyautogui.screenshot()
        full_img_np = np.array(screenshot_full)
        full_img_bgr = cv2.cvtColor(full_img_np, cv2.COLOR_RGB2BGR)
        
        # Draw a rectangle on the full-screen image to indicate the capture region.
        top_left = (region[0], region[1])
        bottom_right = (region[0] + region[2], region[1] + region[3])
        cv2.rectangle(full_img_bgr, top_left, bottom_right, (0, 255, 0), 3)

        # Prepare the image for the model:
        input_image = Image.fromarray(img_np)
        input_tensor = test_transform(input_image).unsqueeze(0)  # Add batch dimension
        
        # Run the model inference.
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = CLASSES[predicted.item()]
        
        # Overlay the prediction text on the display image.
        cv2.putText(display_img, f"Predicted: {predicted_class}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the resulting image.
        cv2.imshow("Real-Time Location Prediction", display_img)
        
        # Exit loop if 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
