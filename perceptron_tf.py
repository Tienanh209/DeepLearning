import tensorflow as tf

# Custom Dense Layer (Modified to return only output)
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim=8):
        super(MyDenseLayer, self).__init__()
        self.W = self.add_weight(shape=[input_dim, output_dim])
        self.b = self.add_weight(shape=[1, output_dim])

    def call(self, inputs):
        z = tf.matmul(inputs, self.W) + self.b
        return tf.nn.relu(z)

# Model with multiple layers
model = tf.keras.Sequential([
    MyDenseLayer(input_dim=3, output_dim=8),          # Custom hidden layer (ReLU)
    tf.keras.layers.Dense(4, activation='relu'),      # Built-in dense layer
    tf.keras.layers.Dense(1, activation='sigmoid')    # Output layer
])

# Hours studied, hours slept, practice tests
student_data = tf.constant([[7.0, 7.0, 4.0]], dtype=tf.float32)

# Run the model
output = model(student_data)

print(f"Passing: {round(output.numpy()[0][0] * 100)}%")
print(f"Failing: {round((1 - output.numpy()[0][0]) * 100)}%")
