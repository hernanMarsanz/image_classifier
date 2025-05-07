import torch
import torchvision.transforms as transforms
from PIL import Image

# --- Configuration ---
MODEL_PATH = "models/vgg16.pth"  # Replace with the actual path to your .pth file
IMAGE_PATH = "media/pc.png"  # Replace with the actual path to your image
# --- End Configuration ---

# Define the VGG-16 model architecture (you might need to adjust based on your .pth)
class VGG16(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Define the image preprocessing steps for VGG16
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_and_preprocess_image(image_path):
    """Loads an image and preprocesses it for VGG16."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img)
        # Add a batch dimension (N, C, H, W)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None

def load_vgg16_model(model_path, num_classes=1000):
    """Loads the VGG16 model and its pretrained weights."""
    model = VGG16(num_classes=num_classes)
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        # Handle different ways weights might be saved
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()  # Set the model to evaluation mode
        return model
    except FileNotFoundError:
        print(f"Error: Model weights not found at {model_path}")
        return None
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        return None

def get_imagenet_classes(file_path="models/imagenet_classes.txt"):
    """Loads the ImageNet class labels from a text file."""
    try:
        with open(file_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Predictions will be raw indices.")
        return None

if __name__ == "__main__":
    # Load and preprocess the image
    print('Start image preprocessing')
    input_tensor = load_and_preprocess_image(IMAGE_PATH)
    print('Image preprocessing done.')

    if input_tensor is not None:
        # Load the VGG16 model
        model = load_vgg16_model(MODEL_PATH)

        if model is not None:
            print('Starting inference.')
            # Make the inference
            with torch.no_grad():
                output = model(input_tensor)

            # Get the predicted class index
            _, predicted_idx = torch.max(output, 1)
            predicted_idx = predicted_idx.item()

            # Load ImageNet class labels (optional)
            imagenet_classes = get_imagenet_classes()
            if imagenet_classes:
                predicted_class = imagenet_classes[predicted_idx]
                print(f"Predicted class: {predicted_class} (Index: {predicted_idx})")
            else:
                print(f"Predicted class index: {predicted_idx}")
            print('Inferencing done.')