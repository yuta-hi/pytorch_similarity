from __future__ import absolute_import

import torch
import numpy as np

def len_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def image_to_tensor(image, batch_size=1):

    assert(image.ndim in [2,3])

    if image.ndim == 2:
        image = np.expand_dims(image, axis=(0)) # channel

    image = np.expand_dims(image, axis=(0)) # batch
    image = np.concatenate([image]*batch_size, axis=0)

    return torch.from_numpy(image).float()
