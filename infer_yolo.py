import os
from ultralytics import YOLO

def main():
    # Path to trained weights
    weights = os.path.join('yolo_train', 'exp', 'weights', 'best.pt')
    # Directory of images to test (change as needed)
    source = os.path.join('data', 'images', 'test')
    # Run inference with visualization on CPU (change device='cpu' to device='0' or device='cuda' for GPU)
    model = YOLO(weights)
    results = model.predict(source=source, save=True, imgsz=640, device='cuda', project='yolo_infer', name='exp', exist_ok=True)
    print('Inference complete. Visualized predictions saved in yolo_infer/exp')

if __name__ == '__main__':
    main()
