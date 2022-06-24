import torch


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