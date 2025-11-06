import os
from ultralytics import YOLO

def main():
    # Path to trained weights and data.yaml
    weights = os.path.join('yolo_train', 'exp', 'weights', 'best.pt')
    data_yaml = os.path.join('data', 'data.yaml')
    # Evaluate on validation set with metrics and visualizations
    model = YOLO(weights)
    results = model.val(data=data_yaml, split='val', save=True, imgsz=640, project='yolo_eval', name='exp', exist_ok=True)
    print('Evaluation complete. Visualizations and metrics saved in yolo_eval/exp')

if __name__ == '__main__':
    main()
