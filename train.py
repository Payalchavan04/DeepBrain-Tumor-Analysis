import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# ---------- SETTINGS ----------
IMAGE_SIZE = 150
batch_size = 16

# 👉 UPDATED PATHS (FIXED)
train_path = "data/train"
test_path = "data/test"

# ---------- DATA LOADING ----------
train_datagen = ImageDataGenerator(
    rescale=1/255.0,         # Convert pixel values to 0–1
    rotation_range=15,       # Slight rotation
    zoom_range=0.1,          # Slight zoom
    horizontal_flip=True,    # Flip images
    validation_split=0.2     # 20% images for validation
)

test_datagen = ImageDataGenerator(rescale=1/255.0)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=batch_size,
    class_mode='categorical',    # 4 classes → categorical
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# ---------- MODEL CREATION ----------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')   # 4 output classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


checkpoint = ModelCheckpoint("models/brain_tumor_model.h5", save_best_only=True, monitor='val_loss')

# ---------- TRAINING ----------
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=[checkpoint]
)

print("Training Completed. Best Model Saved!")


plt.plot(history.history['accuracy'], label="train accuracy")
plt.plot(history.history['val_accuracy'], label="val accuracy")
plt.legend()
plt.show()
