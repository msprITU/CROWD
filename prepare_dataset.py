import os
import numpy as np
from PIL import Image
import h5py
import argparse

def create_h5(name: str, directory: str):
    CLASS_MAP = {
        "PEOPLE": 1, "BICYCLE": 2, "CAR": 3, "VAN": 4,
        "TRUCK": 5, "TRICYCLE": 6, "BUS": 7, "MOTOR": 8
    }
    
    images_dir = os.path.join(directory, "images")
    gt_dir = os.path.join(directory, "gt_files")
    h5_dir = os.path.join(directory, name)
    
    os.makedirs(h5_dir, exist_ok=True)
    unknown_classes = set()

    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(".jpg"):
            continue
            
        img_path = os.path.join(images_dir, img_name)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(gt_dir, txt_name)
        
        if not os.path.exists(txt_path):
            print(f"Warning: No gt file for {img_name}")
            continue

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        height, width = img_np.shape[:2]

        # Prepare kpoint matrix
        kpoint = np.zeros((height, width), dtype=np.uint8)

        # Read gt file
        with open(txt_path, "r") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) < 3:
                    continue
                try:
                    y = int(float(parts[0]))
                    x = int(float(parts[1]))
                    class_name = parts[2]
                except Exception as e:
                    print(f"Error parsing line in {txt_name}: {line}")
                    continue

                class_num = CLASS_MAP.get(class_name)
                if class_num is None:
                    unknown_classes.add(class_name)
                    continue

                if 0 <= y < height and 0 <= x < width:
                    kpoint[y, x] = class_num

        # Save to h5
        h5_name = os.path.splitext(img_name)[0] + ".h5"
        h5_path = os.path.join(h5_dir, h5_name)
        with h5py.File(h5_path, "w") as h5f:
            h5f.create_dataset("image", data=img_np, compression="gzip")
            h5f.create_dataset("kpoint", data=kpoint, compression="gzip")
            
    if unknown_classes:
        print("Unknown classes found:", unknown_classes)

    print(f"[{name}] Dataset converted to h5!")

def create_numpy(train_or_test: str, directory: str): 
    if train_or_test not in ['train', 'test']:
        print("Please only type in 'train' or 'test' as the first parameter.")
        return 0
    
    h5_dir = os.path.join(directory, f'{train_or_test}_h5')
    h5_files = [os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith('.h5')]
    h5_files_np = np.array(h5_files)
    
    output_path = os.path.join(directory, f'visdrone_{train_or_test}.npy')
    
    with open(output_path, 'wb') as f:
        np.save(f, h5_files_np)
    
    print(f"Saved {len(h5_files_np)} file paths to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare VisDrone dataset (h5 and npy)")
    parser.add_argument('--data_dir', type=str, required=True, help="Base directory containing 'train' and 'test' folders")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    # Train 
    if os.path.exists(train_dir):
        print("Processing Train Data...")
        create_h5("train_h5", train_dir)
        create_numpy("train", train_dir)
    else:
        print(f"Train directory not found: {train_dir}")

    # Test 
    if os.path.exists(test_dir):
        print("Processing Test Data...")
        create_h5("test_h5", test_dir)
        create_numpy("test", test_dir)
    else:
        print(f"Test directory not found: {test_dir}")