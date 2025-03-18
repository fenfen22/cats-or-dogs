from tensorflow.keras.preprocessing.image import ImageDataGenerator

def prepare_data():
    train_data = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_data = ImageDataGenerator(rescale=1./255)

    train_generator = train_data.flow_from_directory(
    'data/processed/train',  # Directory with training data
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,
    class_mode='binary'  # Binary classification (dogs vs cats)
    )

    vali_generator = test_data.flow_from_directory(
    'data/processed/vali',  # Directory with training data
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,
    class_mode='binary'  # Binary classification (dogs vs cats)
    )

    test_generator = test_data.flow_from_directory(
    'data/processed/test',  # Directory with training data
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,
    class_mode='binary'  # Binary classification (dogs vs cats)
    )

    return train_generator, vali_generator, test_generator

