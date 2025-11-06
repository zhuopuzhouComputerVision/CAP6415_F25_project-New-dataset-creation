import json
import random
from pathlib import Path
from shutil import copy2
from PIL import Image

# Config
SRC_IMG_DIR = Path('cat_image')
SRC_LABEL_DIR = Path('cat_label')
DST_ROOT = Path('data')
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
RANDOM_SEED = 42

# Output dirs
IMG_TRAIN = DST_ROOT / 'images' / 'train'
IMG_VAL = DST_ROOT / 'images' / 'val'
IMG_TEST = DST_ROOT / 'images' / 'test'
LBL_TRAIN = DST_ROOT / 'labels' / 'train'
LBL_VAL = DST_ROOT / 'labels' / 'val'
LBL_TEST = DST_ROOT / 'labels' / 'test'

DST_ROOT.mkdir(exist_ok=True)
for d in [IMG_TRAIN, IMG_VAL, IMG_TEST, LBL_TRAIN, LBL_VAL, LBL_TEST]:
    d.mkdir(parents=True, exist_ok=True)

def find_classes(label_dir):
    classes = set()
    for f in label_dir.glob('*.json'):
        with open(f, 'r', encoding='utf-8') as jf:
            data = json.load(jf)
            for shape in data.get('shapes', []):
                if 'label' in shape:
                    classes.add(shape['label'])
    return sorted(classes)

def convert_shape_to_yolo(shape, img_w, img_h, class_map):
    if shape.get('shape_type') != 'rectangle':
        return None
    label = shape['label']
    pts = shape['points']
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    x_c = (x_min + x_max) / 2.0 / img_w
    y_c = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    class_id = class_map[label]
    return f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"

def process_one(json_path, class_map):
    with open(json_path, 'r', encoding='utf-8') as jf:
        data = json.load(jf)
    img_rel = data['imagePath'].replace('..\\', '').replace('..//', '')
    img_path = Path(img_rel)
    if not img_path.is_absolute():
        img_path = SRC_IMG_DIR / img_path.name
    if not img_path.exists():
        return None, None, f"Image not found: {img_path}"
    try:
        with Image.open(img_path) as im:
            img_w, img_h = im.size
    except Exception as e:
        return None, None, f"Failed to open image {img_path}: {e}"
    yolo_lines = []
    for shape in data.get('shapes', []):
        yolo = convert_shape_to_yolo(shape, img_w, img_h, class_map)
        if yolo:
            yolo_lines.append(yolo)
    return img_path, yolo_lines, None

def main():
    random.seed(RANDOM_SEED)
    json_files = sorted(SRC_LABEL_DIR.glob('*.json'))
    if not json_files:
        print('No JSON files found in', SRC_LABEL_DIR)
        return
    # Find all classes
    classes = find_classes(SRC_LABEL_DIR)
    class_map = {name: i for i, name in enumerate(classes)}
    print('Classes:', class_map)
    # Shuffle and split
    indices = list(range(len(json_files)))
    random.shuffle(indices)
    n = len(indices)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val
    train_idx = set(indices[:n_train])
    val_idx = set(indices[n_train:n_train+n_val])
    test_idx = set(indices[n_train+n_val:])
    # Process
    stats = {'train': 0, 'val': 0, 'test': 0, 'skipped': 0, 'errors': []}
    for i, json_path in enumerate(json_files):
        img_path, yolo_lines, err = process_one(json_path, class_map)
        if err:
            stats['skipped'] += 1
            stats['errors'].append(f"{json_path.name}: {err}")
            continue
        base = json_path.stem
        if i in train_idx:
            img_dst = IMG_TRAIN / img_path.name
            lbl_dst = LBL_TRAIN / f"{base}.txt"
            stats['train'] += 1
        elif i in val_idx:
            img_dst = IMG_VAL / img_path.name
            lbl_dst = LBL_VAL / f"{base}.txt"
            stats['val'] += 1
        else:
            img_dst = IMG_TEST / img_path.name
            lbl_dst = LBL_TEST / f"{base}.txt"
            stats['test'] += 1
        copy2(img_path, img_dst)
        with open(lbl_dst, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines) + ('\n' if yolo_lines else ''))
    # Write data.yaml and print summary
    yaml = (
        'train: ' + str(IMG_TRAIN.as_posix()) + '\n'
        'val: ' + str(IMG_VAL.as_posix()) + '\n'
        'test: ' + str(IMG_TEST.as_posix()) + '\n'
        'nc: ' + str(len(classes)) + '\n'
        'names: ' + str(classes) + '\n'
    )
    with open(DST_ROOT / 'data.yaml', 'w', encoding='utf-8') as f:
        f.write(yaml)
    print('Done. Train: {}, Val: {}, Test: {}, Skipped: {}'.format(stats['train'], stats['val'], stats['test'], stats['skipped']))
    if stats['errors']:
        print('Errors:')
        for e in stats['errors'][:10]:
            print(' ', e)
        if len(stats['errors']) > 10:
            print('  ...and {} more'.format(len(stats['errors'])-10))
    print('Sample data.yaml:')
    print(yaml)

if __name__ == '__main__':
    main()

