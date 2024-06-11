from google.colab import drive
drive.mount('/content/drive')
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


dataset_dir = '/content/drive/MyDrive/Palm/'


image_size = 32
class_names = ['Non-anemic', 'Anemic']
num_classes = len(class_names)


def load_and_preprocess_image(image_path, image_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size, image_size))
    image = image / 255.0
    return image


def extract_label_from_filename(filename):
    if 'Non-anemic' in filename:
        return 0
    elif 'Anemic' in filename:
        return 1
    else:
        raise ValueError(f"Unknown label in filename: {filename}")


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

labels = to_categorical(labels, num_classes)

X_temp, X_test, y_temp, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Test data shape: {X_test.shape}")
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow_addons.optimizers import AdamW

class PatchExtractor(layers.Layer):
    def __init__(self, patch_size):
        super(PatchExtractor, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

def create_vit_classifier(image_size, patch_size, num_layers, hidden_units, num_heads, projection_dim, mlp_head_units, num_classes):
    num_patches = (image_size // patch_size) ** 2
    inputs = layers.Input(shape=(image_size, image_size, 3))
    patches = PatchExtractor(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(num_layers):

        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)

        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)(x1, x1)

        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = LayerNormalization(epsilon=1e-6)(x2)

        x3 = mlp(x3, hidden_units, dropout_rate=0.1)

        encoded_patches = layers.Add()([x3, x2])

    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = Dropout(0.5)(representation)

    features = mlp(representation, mlp_head_units, dropout_rate=0.5)
    logits = Dense(num_classes)(features)
    model = Model(inputs=inputs, outputs=logits)
    return model


patch_size = 4
num_layers = 8
hidden_units = [128, 64]
num_heads = 4
projection_dim = 64
mlp_head_units = [2048, 1024]


vit_classifier = create_vit_classifier(
    image_size=image_size,
    patch_size=patch_size,
    num_layers=num_layers,
    hidden_units=hidden_units,
    num_heads=num_heads,
    projection_dim=projection_dim,
    mlp_head_units=mlp_head_units,
    num_classes=num_classes
)

vit_classifier.compile(
    optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

vit_classifier.fit(
    X_train, y_train,
    batch_size=64,
    epochs=20,
    validation_data=(X_val, y_val)
)


val_loss, val_acc = vit_classifier.evaluate(X_val, y_val)
print(f'Validation accuracy: {val_acc:.3f}')

test_loss, test_acc = vit_classifier.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.3f}')


predictions = vit_classifier.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)


import matplotlib.pyplot as plt

num_display = 5
plt.figure(figsize=(10, 10))
for i in range(num_display):
    plt.subplot(1, num_display, i + 1)
    plt.imshow(X_test[i])
    plt.title(f'True: {class_names[true_labels[i]]}\nPred: {class_names[predicted_labels[i]]}')
    plt.axis('off')
plt.show()
