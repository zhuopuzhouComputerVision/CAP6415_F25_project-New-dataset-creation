import os
from ultralytics import YOLO

def main():
    # Path to data.yaml
    data_yaml = os.path.join('data', 'data.yaml')
    # Create YOLO model (use yolov8n for speed, change as needed)
    model = YOLO('yolov8n.pt')
    # Train
    model.train(data=data_yaml, epochs=5, imgsz=640, batch=16, project='yolo_train', name='exp', exist_ok=True)

if __name__ == '__main__':
    main()
