from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
model.train(data="./cricket-ball-images3/data.yaml", epochs=100, device = 'mps')  # train the model