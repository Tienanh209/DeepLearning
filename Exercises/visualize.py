import numpy as np
from matplotlib import pyplot as plt

file_path = "datasets/full_numpy_bitmap_ant.npy"

images = np.load(file_path)
train_images = images[:-10]
test_images = images[-10:]

print(images.shape)

avg_images = np.mean(train_images)
print(avg_images)