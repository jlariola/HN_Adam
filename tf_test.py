import tensorflow as tf
from hn_adam_tf import HN_Adam

print("TensorFlow Version:", tf.__version__)

# Create a dummy variable (representing a neural network weight)
dummy_weight = tf.Variable([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)

# Initialize your custom HN_Adam optimizer
optimizer = HN_Adam(learning_rate=0.001)

# Perform a dummy training step to force the optimizer to build its moving averages
with tf.GradientTape() as tape:
    # A simple dummy loss function: sum of the weights squared
    loss = tf.reduce_sum(dummy_weight ** 2)

# Calculate the gradient
gradients = tape.gradient(loss, [dummy_weight])

# Apply the gradient using HN_Adam
optimizer.apply_gradients(zip(gradients, [dummy_weight]))

print("Success! HN_Adam initialized and performed a parameter update.")
print("Updated Weight:\n", dummy_weight.numpy())