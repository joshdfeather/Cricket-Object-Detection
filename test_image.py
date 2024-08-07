from ultralytics import YOLO

# Load the trained model
model = YOLO("/Users/joshfeather/Documents/cbod/runs/detect/train8/weights/best.pt")  # Update with the path to your trained model weights

# Perform evaluation on the test dataset
results = model.val(data="./cricket-ball-images/data.yaml", split='test', device='mps')  # Use 'mps' for Apple Silicon or 'cuda' for NVIDIA GPUs