import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# Step 1: Data Preprocessing for Training and Validation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Load training images from a separate folder
train_generator = train_datagen.flow_from_directory(
    "/home/shusrith/Downloads/aoml-hackathon-1/dataset/train",  # Path to training folder
    target_size=(224, 224),  # Resize all images to 224x224
    batch_size=32,
    class_mode="categorical",
)

# Load validation images from a separate folder
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

validation_generator = validation_datagen.flow_from_directory(
    "/home/shusrith/Downloads/aoml-hackathon-1/dataset/validation",  # Path to validation folder
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
)

# Step 2: Load Pre-trained EfficientNetB5 Model
base_model = EfficientNetB5(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)

# Freeze the base model layers
base_model.trainable = False

# Step 3: Add Custom Layers for Pokémon Classification
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Add global pooling layer
x = Dropout(0.5)(x)  # Add dropout for regularization
x = Dense(128, activation="relu")(x)  # Fully connected layer
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation="softmax")(
    x
)  # Final classification layer

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Step 4: Compile the Model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Step 5: Train the Model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,  # Adjust the number of epochs as needed
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
)

# Step 6: Fine-tune the model (optional)
base_model.trainable = True
fine_tune_at = len(base_model.layers) // 2  # Fine-tune last half of layers

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Re-compile with a lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Fine-tune the model
history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,  # Fine-tuning epochs
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
)

# Step 7: Save the trained model
model.save("pokemon_classifier_efficientnetB5.h5")

# Step 8: Prediction for Test Set

# Load the test set
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    "/home/shusrith/Downloads/aoml-hackathon-1/dataset/test",  # Path to the test folder
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,  # No labels since this is the test set
    shuffle=False,  # Do not shuffle for consistency
)

# Predict
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Get class labels
class_labels = list(train_generator.class_indices.keys())

# Display predictions for test images
for i, filename in enumerate(test_generator.filenames):
    print(f"Image: {filename} -> Predicted: {class_labels[predicted_classes[i]]}")
