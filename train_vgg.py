import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# ---------------- SETTINGS ----------------
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS_PHASE1 = 8
EPOCHS_PHASE2 = 15

# ✅ FIXED PATHS (because file is inside scripts/)
TRAIN_DIR = "../data/train"
VAL_DIR = "../data/test"

MODEL_PATH = "../models/vgg16_brain_tumor.h5"
NUM_CLASSES = 4

# ---------------- DATA GENERATORS ----------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.10,
    height_shift_range=0.10,
    shear_range=0.08,
    zoom_range=0.1,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# 🔍 IMPORTANT: print class index order (for prediction correctness)
print("Class Indices:", train_generator.class_indices)

# ---------------- CLASS WEIGHTS ----------------
labels = train_generator.classes
classes = np.unique(labels)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=labels
)
class_weights = {i: w for i, w in enumerate(class_weights)}
print("Class Weights:", class_weights)

# ---------------- BUILD MODEL ----------------
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_layer)

# ---------------- PHASE 1: FEATURE EXTRACTION ----------------
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=3,
    factor=0.5,
    verbose=1
)

print("Starting Phase 1 training...")
history1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weights
)

# ---------------- PHASE 2: FINE-TUNING ----------------
for layer in base_model.layers[-12:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Starting Phase 2 fine-tuning...")
history2 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE2,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weights
)

model.save(MODEL_PATH)
print("Training completed! Model saved at:", MODEL_PATH)

# ---------------- PLOT RESULTS ----------------
def plot_results(h1, h2):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.legend()
    plt.title("Loss")
    plt.show()

plot_results(history1, history2)
