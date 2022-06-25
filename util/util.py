import torch
import numpy as np
import PIL.Image as Image
import cv2


def tensor_gray(input):
    batch = input.shape[0]
    newinput = input.clone()
    for i in range(0, batch):
        newinput[i, 0, :, :] = input[i, 0, :, :] / 3
        newinput[i, 0, :, :] = newinput[i, 0, :, :] + input[i, 1, :, :] / 3
        newinput[i, 0, :, :] = newinput[i, 0, :, :] + input[i, 2, :, :] / 3
    return newinput[:, 0, :, :]

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def sample(R, L, E, i):
    out = R * L + E
    out_name = "./run/" + str(i) + ".jpg"
    out = out.cpu().detach().numpy()
    out = np.clip(out * 255.0, 0, 255)
    img = Image.fromarray(np.uint8(out.transpose(1, 2, 0)))
    img.save(out_name)

    out_no_noise = R * L
    out_no_noise_name = "./run/" + str(i) + "_no_noise.jpg"
    out_no_noise = out_no_noise.cpu().detach().numpy()
    out_no_noise = np.clip(out_no_noise * 255.0, 0, 255)
    img_no_noise = Image.fromarray(np.uint8(out_no_noise.transpose(1, 2, 0)))
    img_no_noise.save(out_no_noise_name)










