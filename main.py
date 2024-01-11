from ultralytics import YOLO

DATA_DIR = 'data/'
# Load a model
model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='mnist160', epochs=1, imgsz=64)
