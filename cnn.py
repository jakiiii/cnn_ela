import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import tensorflow as tf

np.random.seed(2)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from PIL import Image, ImageChops, ImageEnhance
import PIL
import os
import itertools
from tqdm import tqdm
from sklearn.metrics import classification_report

tf.__version__


# %%
def ELA(img_path, quality=90):
    TEMP = 'ela_' + 'temp.jpg'
    SCALE = 10
    original = Image.open(img_path)
    diff = ""
    try:
        original.save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original, temporary)

    except:

        original.convert('RGB').save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original.convert('RGB'), temporary)

    d = diff.load()
    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            d[x, y] = tuple(k * SCALE for k in d[x, y])
    #     save_path = dataset_path +'ELA_IMAGES/'
    #     diff.save(save_path+'diff.png')
    return diff


# %%
dataset_path = 'CASIA2/'
path_original = 'Au/'
path_tampered = 'Tp/'
# path_mask='CASIA 2 Groundtruth/'
total_original = os.listdir(dataset_path + path_original)
total_tampered = os.listdir(dataset_path + path_tampered)
# total_mask=os.listdir(dataset_path+path_mask)
# %%
pristine_images = []
for i in total_original:
    pristine_images.append(dataset_path + path_original + i)
fake_images = []
for i in total_tampered:
    fake_images.append(dataset_path + path_tampered + i)

len(pristine_images), len(fake_images)
print(1, len(pristine_images), len(fake_images))

image_size = (224, 224)
print(2, image_size)
output_path = '/input/preprocessed-ela-images/'
print(3, output_path)

if not os.path.exists(output_path + "resized_images/"):
    #     os.makedirs(output_path+"resized_images/fake_masks/")
    os.makedirs(output_path + "resized_images/fake_images/")
    os.makedirs(output_path + "resized_images/pristine_images/")
    height = 224
    width = 224
    #     p2=output_path+"resized_images/fake_masks/"
    p1 = output_path + "resized_images/fake_images/"
    p3 = output_path + "resized_images/pristine_images/"
    j = 0
    for fake_image in tqdm(total_tampered):
        try:
            if (j % 1):
                j += 1
                continue
            img = Image.open(dataset_path + path_tampered + fake_image).convert("RGB")
            img = img.resize((height, width), PIL.Image.ANTIALIAS)
            img.save(p1 + fake_image)
            j += 1
        except:
            print("Encountered Invalid File : ", fake_image)

    j = 0
    for pristine_image in tqdm(total_original):
        try:
            if (j % 1):
                j += 1
                continue
            img = Image.open(dataset_path + path_original + pristine_image).convert("RGB")
            img = img.resize((height, width), PIL.Image.ANTIALIAS)
            img.save(p3 + pristine_image)
            j += 1
        except:
            print("Invalid File : ", pristine_image)



else:
    print('images resized,path exists')

resized_fake_image_path = output_path + "resized_images/fake_images/"
resized_pristine_image_path = output_path + "resized_images/pristine_images/"
resized_fake_image = os.listdir(resized_fake_image_path)
resized_pristine_image = os.listdir(resized_pristine_image_path)

print(4, resized_fake_image_path)
print(5, resized_pristine_image_path)
print(6, resized_fake_image)
print(7, resized_pristine_image)

len(resized_fake_image), len(resized_pristine_image)
# %%
ela_images_path = output_path + 'ELA_IMAGES/'
ela_real = ela_images_path + 'Au/'
ela_fake = ela_images_path + 'Tp/'
if not os.path.exists(ela_images_path):
    os.makedirs(ela_images_path)
    os.mkdir(ela_real)
    os.mkdir(ela_fake)
    j = 0
    for i in tqdm(resized_fake_image):
        ELA(resized_fake_image_path + i).save(ela_fake + i)
        j += 1
        if (j == 15000):
            break
    j = 0
    for i in tqdm(resized_pristine_image):
        ELA(resized_pristine_image_path + i).save(ela_real + i)
        j += 1
        if (j == 15000):
            break
else:
    print('Images are already converted to ELA')
# %%
X = []
Y = []
j = 0
for file in tqdm(os.listdir(ela_real)):
    img = Image.open(ela_real + file)
    img = np.array(img)
    X.append(img)
    Y.append(0)
    j += 1
    if (j == 15000):
        break
j = 0
for file in tqdm(os.listdir(ela_fake)):
    img = Image.open(ela_fake + file)
    img = np.array(img)
    X.append(img)
    Y.append(1)
    j += 1
    if (j == 15000):
        break
# %%
X = np.array(X)
X.shape
# %%
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

x_train, x_dev, y_train, y_dev = train_test_split(X, Y, test_size=0.2, random_state=133, shuffle=True)
y_train = to_categorical(y_train, 2)
y_dev = to_categorical(y_dev, 2)
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, AvgPool2D, MaxPool2D, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping


def CNN():
    model = Sequential()
    model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu', input_shape=(224, 224, 3)))
    # model.add(MaxPool2D(pool_size=(2,2)))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
    # model.add(MaxPool2D(pool_size=(2,2)))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=2, activation='softmax'))
    return model


model1 = CNN()
model1.summary()
epochs = 30
batch_size = 32
init_lr = 1e-4

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=init_lr, decay=init_lr / epochs)

model1.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6, verbose=1, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.22, patience=6, verbose=1,
                                              min_delta=0.0001, min_lr=0.0001)
# %%
hist = model1.fit(x_train, y_train,
                  epochs=epochs,
                  validation_data=(x_dev, y_dev),
                  callbacks=[early_stop, reduce_lr],
                  verbose=1, shuffle=True)

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'])
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("Loss")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'])
plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


Y_pred = model1.predict(x_dev)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(y_dev, axis=1)
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
plot_confusion_matrix(confusion_mtx, classes=range(2))

print(classification_report(Y_true, Y_pred_classes))
