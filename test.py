import cv2
import numpy as np
import torch
import tensorflow as tf
from matplotlib import pyplot as plt
from util.loss import *

# img = './imgs/01.jpg'
#
# img = cv2.imread(img)
# img = cv2.resize(img, (400, 300))
# cv2.imshow('1', img)

# print(img.shape)
#
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#
# img_gray_plt = np.concatenate((np.expand_dims(img_gray, axis=2),
#                                np.expand_dims(img_gray, axis=2),
#                                np.expand_dims(img_gray, axis=2)), axis=-1)
# # cv2.imshow('gray', img_gray)
#
# # img_2d = np.zeros((300, 400), dtype=np.float)
# img_2d = (img[:, :, 0] // 3 + img[:, :, 1] // 3 + img[:, :, 2] // 3)
# img_gray_plt1 = np.concatenate((np.expand_dims(img_2d, axis=2),
#                                np.expand_dims(img_2d, axis=2),
#                                np.expand_dims(img_2d, axis=2)), axis=-1)
# # cv2.imshow('gray', img_2d)
#
#
# plt.imshow(img)
# # plt.imshow(img_gray_plt1)
# plt.show()


# k = tf.constant([[0, 0], [-1, 1]], tf.float32)
# print(k)

# kp = torch.tensor([[[[1, 3, 5], [2, 4, 6], [3, 5, 7]]]], dtype=torch.float32)
# print(kp.shape)
# loss = smooth(kp, kp)
# print(loss)
# print(kp.shape[0])
# print(kp - 1)
#
#
#
# k = tf.reshape(k, [2, 2, 1, 1])
# print(k)
#
# kp = torch.reshape(kp, [2, 2, 1, 1])
# print(kp)
#
#
#
# k = tf.transpose(k, [1, 0, 2, 3])
# print(k)
# kp = torch.transpose(kp, 0, 1)
# print(kp)


k = torch.randn([4, 1, 75, 100])
deconv = torch.nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1)
k = deconv(k)
print(k.shape)












