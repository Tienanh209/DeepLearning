import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
import numpy as np

# Load and preprocess IMDB dataset
vocab_size = 10000  # Limit to top 10,000 words
max_length = 200    # Max sequence length
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

# Pad sequences to ensure uniform length
train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, maxlen=max_length)
test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, maxlen=max_length)

# Build the RNN model
model = models.Sequential([
    layers.Embedding(vocab_size, 100, input_length=max_length),
    layers.SimpleRNN(64),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
model.fit(train_data, train_labels, epochs=20, batch_size=64, validation_split=0.2)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print(f'Test accuracy: {test_accuracy:.4f}')