import torch
import numpy as np
import PIL.Image as Image
import torchvision.transforms as transform
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


def RGB2BGR(input):
    output = input
    output[0, :, :] = input[2, :, :]
    output[2, :, :] = input[0, :, :]
    return output


def sample(R, L, E, i, img):
    unloader = transform.ToPILImage()

    input = img
    input_name = "./run/" + str(i) + "_pre.jpg"
    input = input.cpu().detach().numpy()
    input = np.clip(input * 255.0, 0, 255).astype(np.uint8)
    # input = RGB2BGR(input)
    # print(input.shape)
    # img = Image.fromarray(np.uint8(input))
    img = unloader(input.transpose([1, 2, 0]))
    img.save(input_name)

    out = R * L + E
    out_name = "./run/" + str(i) + "_with_noise.jpg"
    out = out.cpu().detach().numpy()
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    # img = Image.fromarray(np.uint8(out))
    img = unloader(out.transpose([1, 2, 0]))
    img.save(out_name)

    out_no_noise = R * L
    out_no_noise_name = "./run/" + str(i) + "_no_noise.jpg"
    out_no_noise = out_no_noise.cpu().detach().numpy()
    out_no_noise = np.clip(out_no_noise * 255.0, 0, 255).astype(np.uint8)
    # img_no_noise = Image.fromarray(np.uint8(out_no_noise))
    img_no_noise = unloader(out_no_noise.transpose([1, 2, 0]))
    img_no_noise.save(out_no_noise_name)










