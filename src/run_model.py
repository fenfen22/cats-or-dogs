from tensorflow.keras import layers
import tensorflow as tf
from data_handler import prepare_data
from model import CNN

import matplotlib.pyplot as plt



def run():
    train_generator, vali_generator, test_generator = prepare_data()

    input_shape = (224, 224, 3)  # Example input shape, adjust as needed
    inputs = layers.Input(shape=input_shape)  # Define the input layer

    ## inistantiate the model
    model = CNN()

    # Connect the model to the input layer
    output = model(inputs)

    # Create the final model by specifying input and output
    model = tf.keras.Model(inputs=inputs, outputs=output)

    # Compile the model
    model.compile(optimizer='adam',
                loss='binary_crossentropy',  # Use 'categorical_crossentropy' for multi-class
                metrics=['accuracy'])

    # Print the model summary
    model.summary()

    # Train the model (assuming train_generator and validation_generator are defined)
    history = model.fit(
        train_generator,
        # steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=10,
        validation_data=vali_generator,
        # validation_steps=vali_generator.samples // vali_generator.batch_size
    )
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def main():
    run()

if __name__ == "__main__":
    main()
