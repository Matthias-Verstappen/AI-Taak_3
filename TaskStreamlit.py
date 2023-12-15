import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import image_dataset_from_directory

NUM_CLASSES = 5
IMG_SIZE = 64
HEIGHT_FACTOR = 0.2
WIDTH_FACTOR = 0.2

def build_model(learning_rate=0.001):
    model = tf.keras.Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(HEIGHT_FACTOR, WIDTH_FACTOR),
        layers.RandomZoom(0.2),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def load_data():
    batch_size = 32
    image_size = (64, 64)
    validation_split = 0.2

    train_ds = image_dataset_from_directory(
        directory='training_set',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset='training',
        seed=123
    )

    validation_ds = image_dataset_from_directory(
        directory='training_set',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset='validation',
        seed=123
    )

    test_ds = image_dataset_from_directory(
        directory='testing_set',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size
    )

    return train_ds, validation_ds, test_ds

def train_model(model, train_ds, validation_ds, epochs, patience):
    callback = EarlyStopping(monitor='val_loss', patience=patience)
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        steps_per_epoch=len(train_ds),
        validation_steps=len(validation_ds),
        epochs=epochs,
        callbacks=[callback]
    )

    return history

def main():
    st.title("Deep learning algorithm")
    st.info("This algorithm classifies five dog breeds. You can change some of the settings on the left to see how that influences the algorithm.")

    st.sidebar.header("Model Training Controls")
    epochs = st.sidebar.slider("Number of Epochs", 1, 500, 10)
    patience = st.sidebar.slider("EarlyStopping Patience", 1, 50, 10)
    st.sidebar.caption("Recommended: 6")
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, step=0.0001, format="%f")
    st.sidebar.caption("Recommended: 0.001")


    train_ds, validation_ds, test_ds = load_data()
    model = build_model(learning_rate)

    if st.button("Train Model"):
        progress_text = st.empty()
        progress_text.text("Training in progress...")
        history = train_model(model, train_ds, validation_ds, epochs, patience)

        progress_text.text(f"Training complete! Number of epochs run: {len(history.history['loss'])}")

        # Plotting the loss and accuracy curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.plot(history.history['loss'], label='training loss')
        ax1.plot(history.history['val_loss'], label='validation loss')
        ax1.set_title('Loss curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.plot(history.history['accuracy'], label='training accuracy')
        ax2.plot(history.history['val_accuracy'], label='validation accuracy')
        ax2.set_title('Accuracy curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        st.pyplot(fig)

        # Evaluate on the test set
        test_loss, test_acc = model.evaluate(test_ds)
        st.text(f'Test accuracy: {test_acc}')

if __name__ == "__main__":
    main()
