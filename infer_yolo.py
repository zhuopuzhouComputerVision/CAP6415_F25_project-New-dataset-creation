import os
from ultralytics import YOLO

def main():
    # Path to trained weights
    weights = os.path.join('yolo_train', 'exp', 'weights', 'best.pt')
    # Directory of images to test (change as needed)
    source = os.path.join('data', 'images', 'test')
    # Run inference with visualization
    model = YOLO(weights)
    results = model.predict(source=source, save=True, imgsz=640, project='yolo_infer', name='exp', exist_ok=True)
    print('Inference complete. Visualized predictions saved in yolo_infer/exp')

if __name__ == '__main__':
    main()
