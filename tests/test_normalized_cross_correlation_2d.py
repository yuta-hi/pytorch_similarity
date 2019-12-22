from __future__ import absolute_import

from torch.autograd import Variable
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1

from torch_similarity.modules import NormalizedCrossCorrelation
from _helper import image_to_tensor
from _helper import len_trainable_params

def test_normalized_cross_correlation_2d(image_a, image_b, out):

    tensor_a = Variable(image_to_tensor(image_a), requires_grad=True)
    tensor_b = Variable(image_to_tensor(image_b), requires_grad=True)

    model = NormalizedCrossCorrelation(return_map=True)
    print(len_trainable_params(model))

    gc, gc_map = model(tensor_a, tensor_b)
    gc.backward()

    fig = plt.figure(figsize=(16,9))
    ax = plt.subplot(1,3,1)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    im = ax.imshow(image_a, cmap='gray')
    fig.colorbar(im, cax=cax)
    ax.set_title('x')

    ax = plt.subplot(1,3,2)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    im = ax.imshow(image_b, cmap='gray')
    fig.colorbar(im, cax=cax)
    ax.set_title('y')

    ax = plt.subplot(1,3,3)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    im = ax.imshow(gc_map.data.numpy()[0,0], cmap='jet_r')
    fig.colorbar(im, cax=cax)
    im.set_clim(-1e-5, 1e-5)
    ax.set_title('NCC(x,y): %.4f' % gc.data.numpy())
    plt.tight_layout()
    plt.savefig('test_normalized_cross_correlation_2d_%s.png' % out)
    plt.close()

if __name__ == '__main__':

    import cv2
    image = cv2.imread('lenna.png')

    test_normalized_cross_correlation_2d(image[:,100:,0], image[:,100:,0], 'match')
    test_normalized_cross_correlation_2d(image[:,:-100,0], image[:,100:,0], 'unmatch')
