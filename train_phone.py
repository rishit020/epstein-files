from ultralytics import YOLO
import yaml

# Load and fix the data.yaml to use absolute paths
import os
data_yaml_path = os.path.abspath('data/phone/data.yaml')

# Load the yaml
with open(data_yaml_path, 'r') as f:
    data = yaml.safe_load(f)

# Update paths to be absolute
data['path'] = os.path.abspath('data/phone')
data['train'] = 'train/images'
data['val'] = 'valid/images'
data['test'] = 'test/images'

# Save updated yaml
updated_yaml = 'data/phone/data_updated.yaml'
with open(updated_yaml, 'w') as f:
    yaml.dump(data, f)

print("Dataset config:")
print(f"  Train: {data['path']}/train/images")
print(f"  Val: {data['path']}/valid/images")
print(f"  Classes: {data['names']}")
print(f"  Num classes: {data['nc']}")

# Load YOLOv8 nano model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data=updated_yaml,
    epochs=50,
    imgsz=640,
    batch=8,
    name='phone_detector',
    patience=10,
    device='cpu',
    workers=2,
    cache=False,
    pretrained=True,
    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.0005,
    verbose=True,
    save=True,
    save_period=10,
)

print("Training complete!")
print(f"Best model saved at: runs/detect/phone_detector/weights/best.pt")
