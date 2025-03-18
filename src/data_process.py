import os
import shutil
import random


def move_images(image_list, dest_dir, class_dir, class_name):
    for img in image_list:
        shutil.move(os.path.join(class_dir, img), os.path.join(dest_dir, class_name, img))


def split_images(raw_dir: str, class_name: str, train_dir: str, test_dir : str, vali_dir : str):
    class_dir = os.path.join(raw_dir, class_name)
    all_imgs = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
    random.shuffle(all_imgs)
    train_size = int(0.7*len(all_imgs))
    vali_size = int(0.2*len(all_imgs))
    test_size = len(all_imgs) - train_size - vali_size

    train_imgs = all_imgs[:train_size]
    vali_imgs = all_imgs[train_size:train_size+vali_size]
    test_imgs = all_imgs[vali_size+train_size:]

    # Move the images
    move_images(train_imgs, train_dir, class_dir, class_name)
    move_images(vali_imgs, vali_dir, class_dir, class_name)
    move_images(test_imgs, test_dir, class_dir, class_name)

    print(f"Moved {len(train_imgs)} {class_name} images to train.")
    print(f"Moved {len(vali_imgs)} {class_name} images to validation.")
    print(f"Moved {len(test_imgs)} {class_name} images to test.")


def raw_data_split(raw_dir: str, processed_dir: str):
    train_dir = os.path.join(processed_dir, 'train')
    test_dir = os.path.join(processed_dir, 'test')
    vali_dir = os.path.join(processed_dir, 'vali')

    for subdir in [train_dir, vali_dir, test_dir]:
        os.makedirs(os.path.join(subdir, 'cats'), exist_ok=True)
        os.makedirs(os.path.join(subdir, 'dogs'), exist_ok=True)
    
    classes = ['cats', 'dogs']
    for class_name in classes:
        split_images(raw_dir, class_name, train_dir, test_dir, vali_dir)


def main():
    raw_data_split('data/raw/', 'data/processed')

if __name__ == "__main__":
    main()