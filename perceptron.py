import tensorflow as tf
import numpy as np

# Define the custom dense layer
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim=4):  # More units
        super(MyDenseLayer, self).__init__()
        self.W = self.add_weight(shape=[input_dim, output_dim])
        self.b = self.add_weight(shape=[1, output_dim])

    def call(self, inputs):
        z = tf.matmul(inputs, self.W) + self.b
        output = tf.math.sigmoid(z)
        return output  # Only return the output for chaining layers

input_dim = 3

# Create a model with multiple layers
model = tf.keras.Sequential([
    MyDenseLayer(input_dim=input_dim, output_dim=4),  # Custom hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid')    # Output layer
])

student_data = np.array([[7.0, 4.0, 4.0]], dtype=np.float32)
student_data = tf.constant(student_data)

# Run the model
output = model(student_data)

print(f"Passing: {round(output.numpy()[0][0] * 100)}%")
print(f"Failing: {round((1 - output.numpy()[0][0]) * 100)}%")
