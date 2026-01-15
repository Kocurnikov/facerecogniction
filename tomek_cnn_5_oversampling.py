import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, regularizers
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "datasets/balanced/train",
    seed=42,
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "datasets/original/val",
    seed=42,
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "datasets/original/test",
    seed=42,
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32
)

num_classes = len(train_dataset.class_names)
print(f"Liczba klas: {num_classes}")

# normalizacja pikseli
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = train_dataset.map(normalize_img).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.map(normalize_img).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.map(normalize_img).prefetch(tf.data.AUTOTUNE)


class CNNModel(keras.Model):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                                    kernel_regularizer=regularizers.l2(0.001))
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.dropout1 = layers.Dropout(0.3)
        
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                    kernel_regularizer=regularizers.l2(0.001))
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.dropout2 = layers.Dropout(0.3)
        
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                    kernel_regularizer=regularizers.l2(0.001))
        self.bn3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D((2, 2))
        
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation='relu',
                                   kernel_regularizer=regularizers.l2(0.001))
        self.out = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        x = self.dropout2(x, training=training)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = self.dense1(x)
        return self.out(x)


model = CNNModel(num_classes)
model.build((None, 48, 48, 1))
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=40,
    callbacks=[early_stopping, reduce_lr]
)

print("Model training complete...")

# sprawdzenie walidacji
val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# sprawdzenie test
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.title('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend(loc='upper right')

plt.show()