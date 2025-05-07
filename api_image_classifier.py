import requests
import json
from PIL import Image
import io

# --- Configuration ---
TORCHSERVE_INFERENCE_URL = "http://localhost:8080/predictions/your_model_name"
IMAGE_PATH = "path/to/your/image.jpg"  # Replace with the actual path to your image
# --- End Configuration ---

def classify_image(image_path, inference_url):
    """
    Sends an image to the TorchServe server for classification and returns the prediction.
    """
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        headers = {'Content-Type': 'image/jpeg'}  # Adjust content type based on your image format
        response = requests.post(inference_url, headers=headers, data=image_bytes)
        response.raise_for_status()  # Raise an exception for bad status codes

        try:
            prediction = response.json()
            return prediction
        except json.JSONDecodeError:
            print(f"Error decoding JSON response: {response.text}")
            return None

    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error sending request to TorchServe: {e}")
        return None

if __name__ == "__main__":
    prediction_result = classify_image(IMAGE_PATH, TORCHSERVE_INFERENCE_URL)

    if prediction_result:
        print("TorchServe Prediction:")
        print(json.dumps(prediction_result, indent=2))