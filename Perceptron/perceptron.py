import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

X_train = tf.constant(X_train, dtype=tf.float32)
y_train = tf.constant(y_train, dtype=tf.int32)
X_test = tf.constant(X_test, dtype=tf.float32)
y_test = tf.constant(y_test, dtype=tf.int32)

# Define model using built-in layers
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Loss, optimizer, metrics
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# Training Loop
epochs = 100
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        logits = model(X_train, training=True)
        loss = loss_fn(y_train, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    accuracy_metric.update_state(y_train, logits)
    acc = accuracy_metric.result().numpy()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.numpy():.4f} - Accuracy: {acc:.4f}")
    accuracy_metric.reset_state()

# Evaluate on test set
test_logits = model(X_test, training=False)
test_preds = tf.argmax(test_logits, axis=1)
test_acc = tf.reduce_mean(tf.cast(test_preds == tf.cast(y_test, tf.int64), tf.float32))
print(f"\nTest Accuracy: {test_acc.numpy():.2f}")
