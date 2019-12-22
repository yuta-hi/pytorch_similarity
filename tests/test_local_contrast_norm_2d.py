from __future__ import absolute_import

from torch.autograd import Variable
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1

from torch_similarity.modules import LocalContrastNorm2d
from _helper import image_to_tensor
from _helper import len_trainable_params

def test_local_contrast_norm_2d(image):

    tensor = Variable(image_to_tensor(image), requires_grad=True)

    model = LocalContrastNorm2d()
    print(len_trainable_params(model))

    lcn = model(tensor)

    fig = plt.figure(figsize=(16,9))
    ax = plt.subplot(1,2,1)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    im = ax.imshow(image, cmap='gray')
    fig.colorbar(im, cax=cax)
    ax.set_title('x')

    ax = plt.subplot(1,2,2)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    im = ax.imshow(lcn.data.numpy()[0,0], cmap='gray')
    fig.colorbar(im, cax=cax)
    ax.set_title('LCN(x)')

    plt.tight_layout()
    plt.savefig('test_local_contrast_norm_2d.png')
    plt.close()

if __name__ == '__main__':

    import cv2
    image = cv2.imread('lenna.png')

    test_local_contrast_norm_2d(image[:,:,0])
