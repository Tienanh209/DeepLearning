import cv2
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

img = cv2.imread('pic/mxm.jpg')
img = cv2.resize(img, (200,200))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
# print(img_gray.shape)

class Conv2d:
    def __init__(self, input, numOfKernel=8, kernelSize=3, padding=0, stride=1):
        self.input = np.pad(input, ((padding, padding), (padding, padding)), 'constant')
        # self.height, self.width = input.shape
        self.kernel = np.random.randn(numOfKernel, kernelSize, kernelSize)
        self.stride = stride
        # print(self.kernel.shape)

        # tao mang de chua
        self.results = np.zeros((int((self.input.shape[0] - self.kernel.shape[1])/self.stride) + 1,
                                 int((self.input.shape[1] - self.kernel.shape[2])/self.stride) + 1,
                                 self.kernel.shape[0]))

    # roi = region of interesting
    def getROI(self):
        for row in range(int((self.input.shape[0] - self.kernel.shape[1])/self.stride) + 1):
            for col in range(int((self.input.shape[1] - self.kernel.shape[2])/self.stride) + 1):
                roi = self.input[row*self.stride: row*self.stride + self.kernel.shape[1],
                                 col*self.stride: col*self.stride + self.kernel.shape[2]]
                yield row, col, roi

    def operate(self):
        for layer in range(self.kernel.shape[0]):
            for row, col, roi in self.getROI():
                self.results[row, col, layer] = np.sum(roi * self.kernel[layer])

        return self.results

class Relu:
    def __init__(self, input):
        self.input = input
        self.results = np.zeros((self.input.shape[0],
                                 self.input.shape[1],
                                 self.input.shape[2]))
    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(self.input.shape[0]):
                for col in range(self.input.shape[1]):
                    self.results[row, col, layer] = 0 \
                        if self.input[row, col, layer] < 0 \
                        else self.input[row, col, layer]
        return self.results

class LeakyRelu:
    def __init__(self, input):
        self.input = input
        self.results = np.zeros((self.input.shape[0],
                                 self.input.shape[1],
                                 self.input.shape[2]))
    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(self.input.shape[0]):
                for col in range(self.input.shape[1]):
                    self.results[row, col, layer] = 0.1*self.input[row, col, layer] \
                        if self.input[row, col, layer] < 0 \
                        else self.input[row, col, layer]
        return self.results

class MaxPooling:
    def __init__(self, input, poolingSize=2):
        self.input = input
        self.poolingSize = poolingSize
        self.results = np.zeros((int((self.input.shape[0])/self.poolingSize),
                                 int((self.input.shape[1])/self.poolingSize),
                                 self.input.shape[2]))
    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(int((self.input.shape[0])/self.poolingSize)):
                for col in range(int((self.input.shape[1])/self.poolingSize)):
                    self.results[row, col, layer] = np.max(self.input[row*self.poolingSize: row*self.poolingSize+ self.poolingSize,
                                                            col*self.poolingSize: col*self.poolingSize+ self.poolingSize, layer])
        return self.results
    
class Softmax:
    def __init__(self, input, nodes):
        self.input = input
        self.nodes = nodes
        # y = w0 + w(i) * x
        self.flatten = self.input.flatten()
        # print(self.flatten.shape)
        self.weights = np.random.randn(self.flatten.shape[0])/self.flatten.shape[0] # flatten
        self.bias = np.random.randn(nodes)

    def operate(self):
        totals = np.dot(self.flatten, self.weights) + self.bias
        exp = np.exp(totals)
        print(exp.shape)
        return exp/sum(exp)


img_gray_conv2d = Conv2d(img_gray, 16, 3, padding=2, stride=1).operate()
img_gray_conv2d_relu = Relu(img_gray_conv2d).operate()
img_gray_conv2d_leakyrelu = LeakyRelu(img_gray_conv2d).operate()
img_gray_conv2d_leakyrelu_maxpooling = MaxPooling(img_gray_conv2d_leakyrelu, 3).operate()

softmax = Softmax(img_gray_conv2d_leakyrelu_maxpooling, 10).operate()
print(softmax)

# fig = plt.figure(figsize=(10, 10))
# for i in range(16):
#     plt.subplot(4, 4, i+1)
#     plt.imshow(img_gray_conv2d_leakyrelu_maxpooling[:, :, i], cmap='gray')
#     plt.axis('off')
# plt.savefig('img_gray_conv2d_leakyrelu_maxpooling.jpg')
# plt.show()

# for i in range(9):
#     conv2d = Conv2d(img_gray, 3, padding=2, stride=i+1)
#     img_gray_conv2d = conv2d.operate()
#     conv2d_relu = Relu(img_gray_conv2d)
#     img_gray_conv2d_relu = conv2d_relu.operate()
#
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(img_gray_conv2d_relu, cmap='gray')
#     plt.axis('off')
#
# plt.show()

# conv2d = Conv2d(img_gray, 3, 1)
# img_gray_conv2d = conv2d.operate()
# conv2d_relu = Relu(img_gray_conv2d)
# img_gray_conv2d_relu = conv2d_relu.operate()
#
# plt.imshow(img_gray_conv2d_relu, cmap='gray')
# plt.show()
