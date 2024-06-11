from google.colab import drive
drive.mount('/content/drive')

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

dataset_dir = '/content/drive/MyDrive/Palm/'
image_size = 32
class_names = ['Non-anemic', 'Anemic']
num_classes = len(class_names)


def load_and_preprocess_image(image_path, image_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size, image_size))
    image = image / 255.0
    return image

images = []
labels = []

for image_name in os.listdir(dataset_dir):
    image_path = os.path.join(dataset_dir, image_name)
    if os.path.isfile(image_path):
        image = load_and_preprocess_image(image_path, image_size)
        images.append(image)
        label = extract_label_from_filename(image_name)
        labels.append(label)

images = np.array(images)
labels = np.array(labels)


X_temp, X_test, y_temp, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

=
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val))


test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.3f}')
