from ultralytics import YOLO

# Load pre-trained model
model = YOLO("yolov8n.pt")

# Train model
model.train(data="data.yaml", epochs=50, imgsz=640)
