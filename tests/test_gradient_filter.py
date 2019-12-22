import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from torch_similarity.functional import spatial_filter_nd
from torch_similarity._helper import gradient_kernel_1d
from torch_similarity._helper import gradient_kernel_2d
from torch_similarity._helper import gradient_kernel_3d

from chainer_bcnn.data import save_image

def test_gradient_filter_1d(method):

    x = np.linspace(-10, 10, 20)

    kernel_1d = gradient_kernel_1d(method)
    kernel_3d = kernel_1d.reshape(1, 1, *kernel_1d.shape)
    kernel_3d = torch.Tensor(kernel_3d).float()

    dst = spatial_filter_nd(torch.Tensor(x.reshape(1,1,*x.shape)), kernel_3d)
    dst = dst.data.numpy()

    plt.figure()
    plt.plot(x)
    plt.plot(dst[0,0])
    plt.savefig('test_gradient_filter_1d.png')

def test_gradient_filter_2d(method):

    out = 'test_gradient_filter_2d'
    os.makedirs(out, exist_ok=True)

    x = np.linspace(-10, 10, 20)
    y = np.linspace(-10, 10, 20)
    X, Y = np.meshgrid(x, y, indexing='ij')

    for name, src in zip(['X', 'Y'], [X, Y]):
        for axis in range(2):
            kernel_2d = gradient_kernel_2d(method, axis=axis)
            kernel_4d = kernel_2d.reshape(1, 1, *kernel_2d.shape)
            kernel_4d = torch.Tensor(kernel_4d).float()

            dst = spatial_filter_nd(torch.Tensor(src.reshape(1,1,*src.shape)), kernel_4d)
            dst = dst.data.numpy()

            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(src)
            plt.subplot(1,2,2)
            plt.imshow(dst[0,0])
            plt.savefig(os.path.join(out, '%s_grad_%d.png' % (name, axis)))

def test_gradient_filter_3d(method):

    out = 'test_gradient_filter_3d'
    os.makedirs(out, exist_ok=True)

    x = np.linspace(-10, 10, 20)
    y = np.linspace(-10, 10, 20)
    z = np.linspace(-10, 10, 20)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    for name, src in zip(['X', 'Y', 'Z'], [X, Y, Z]):

        save_image(os.path.join(out, '%s.mha' % name), src)

        for axis in range(3):
            kernel_3d = gradient_kernel_3d(method, axis=axis)
            kernel_5d = kernel_3d.reshape(1, 1, *kernel_3d.shape)
            kernel_5d = torch.Tensor(kernel_5d).float()

            dst = spatial_filter_nd(torch.Tensor(src.reshape(1,1,*src.shape)), kernel_5d)
            dst = dst.data.numpy()

            save_image(os.path.join(out, '%s_grad_%d.mha' % (name, axis)), dst[0,0])

if __name__ == '__main__':
    test_gradient_filter_1d(method='default')
    test_gradient_filter_2d(method='default')
    test_gradient_filter_3d(method='default')
