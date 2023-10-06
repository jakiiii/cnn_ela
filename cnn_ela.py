import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, AvgPool2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from PIL import Image, ImageChops
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

np.random.seed(2)


def ela_image(img_path, quality=90):
    """Generate ELA image."""
    TEMP = 'ela_temp.jpg'
    SCALE = 10
    original = Image.open(img_path)
    try:
        original.save(TEMP, quality=quality)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original, temporary)
    except:
        original.convert('RGB').save(TEMP, quality=quality)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original.convert('RGB'), temporary)

    d = diff.load()
    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            d[x, y] = tuple(k * SCALE for k in d[x, y])
    return diff


def preprocess_images(dataset_path, path_original, path_tampered):
    """Preprocess images and return paths."""
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    total_original = [f for f in os.listdir(dataset_path + path_original) if os.path.splitext(f)[1].lower() in valid_extensions]
    total_tampered = [f for f in os.listdir(dataset_path + path_tampered) if os.path.splitext(f)[1].lower() in valid_extensions]

    pristine_images = [dataset_path + path_original + i for i in total_original]
    fake_images = [dataset_path + path_tampered + i for i in total_tampered]

    return pristine_images, fake_images


def load_and_preprocess_image(image_path):
    """Load and preprocess an image from its path."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    return img_array


def cnn_model():
    """Define the CNN model."""
    model = Sequential()
    model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu', input_shape=(224, 224, 3)))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=2, activation='softmax'))
    return model


def plot_metrics(history):
    """Plot training metrics."""
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Accuracy")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'])
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'test'])
    plt.show()


def main():
    # Paths and preprocessing
    dataset_path = '/home/jaki/Dev/cnn_ela/CASIA2/'
    path_original = 'Au/'
    path_tampered = 'Tp/'
    pristine_images, fake_images = preprocess_images(dataset_path, path_original, path_tampered)

    # Labeling the images
    pristine_labels = [0] * len(pristine_images)
    fake_labels = [1] * len(fake_images)

    # Combining the images and labels
    all_images = pristine_images + fake_images
    all_labels = pristine_labels + fake_labels

    # Convert image paths to actual image data
    x_train, x_dev, y_train, y_dev = train_test_split(all_images, all_labels, test_size=0.2, random_state=133,
                                                      shuffle=True)
    x_train = np.array([load_and_preprocess_image(img_path) for img_path in x_train])
    x_dev = np.array([load_and_preprocess_image(img_path) for img_path in x_dev])

    # Normalize the image data to [0, 1] range
    x_train = x_train.astype('float32') / 255.0
    x_dev = x_dev.astype('float32') / 255.0

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 2)
    y_dev = to_categorical(y_dev, 2)

    # Model definition, compilation, and training
    model = cnn_model()
    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_accuracy', patience=6, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.22, patience=6, verbose=1, min_delta=0.0001,
                                  min_lr=0.0001)
    history = model.fit(x_train, y_train, epochs=30, validation_data=(x_dev, y_dev), callbacks=[early_stop, reduce_lr],
                        verbose=1, shuffle=True)

    # Plot metrics
    plot_metrics(history)

    # Evaluation
    Y_pred = model.predict(x_dev)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(y_dev, axis=1)
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    # Assuming you have a function or method to plot the confusion matrix
    # plot_confusion_matrix(confusion_mtx, classes=range(2))
    print(classification_report(Y_true, Y_pred_classes))


if __name__ == "__main__":
    main()

from PIL.Image import ANTIALIAS

from PIL import Image

image = Image.ANTIALIAS
