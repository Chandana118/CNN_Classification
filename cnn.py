# ================================
# 1. Install Required Libraries
# ================================
!pip install -q tensorflow tensorflow-datasets

# ================================
# 2. Import Libraries
# ================================
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

print("TensorFlow Version:", tf.__version__)

# ================================
# 3. Load Dataset (Auto Download)
# ================================
(train_data, test_data), info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True
)

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5

# ================================
# 4. Preprocess Images
# ================================
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

train_data = train_data.map(preprocess)
test_data = test_data.map(preprocess)

train_data = train_data.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_data = test_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ================================
# 5. Build CNN Model
# ================================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# ================================
# 6. Compile Model
# ================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================================
# 7. Train Model
# ================================
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data
)

# ================================
# 8. Evaluate Model
# ================================
loss, accuracy = model.evaluate(test_data)
print("\nTest Accuracy:", accuracy)

# ================================
# 9. Plot Accuracy Graph
# ================================
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("CNN Accuracy")
plt.show()

# ================================
# 10. Save Model
# ================================
model.save("animals_cnn_model.h5")
print("Model Saved Successfully!")
