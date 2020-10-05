"""
Source:
https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/gradcam.py
Modified by Leon Chen
"""

from PIL import Image
import numpy as np
import torch
import matplotlib.cm as mpl_color_map
import copy
import torch.nn as nn
import torch.nn.functional as F

import vis.turbo


def apply_colormap_on_image(x_orig, activation, colormap_name):
    """
    Apply heatmap on image
    args:
        x_orig (numpy arr): Original input
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    returns:
        no_trans_heatmap (PIL Image)
        heatmap_on_image (PIL Image)
    """
    org_im = Image.fromarray(x_orig)

    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert("RGBA"))
    heatmap_on_image = Image.alpha_composite(
        heatmap_on_image, heatmap.resize(org_im.size)
    )
    return no_trans_heatmap, heatmap_on_image


class CamExtractor:
    """
    Extracts cam features from the model
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass(self, x):
        """
        Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        features = nn.Sequential(*list(self.model.children())[:-1])
        x = features(x)

        x.register_hook(self.save_gradient)
        conv_output = x
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)

        return conv_output, x


class GradCam:
    """
        Produces class activation map
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model)

    def generate_cam(self, input_image, original_image, target_class):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        if torch.cuda.is_available():
            one_hot_output = one_hot_output.cuda()
        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients
        weights = np.mean(
            guided_gradients, axis=(1, 2)
        )  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = (
            np.uint8(
                Image.fromarray(cam).resize(
                    (input_image.shape[2], input_image.shape[3]), Image.ANTIALIAS
                )
            )
            / 255
        )
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))

        heatmap, heatmap_on_image = apply_colormap_on_image(
            original_image, cam, "turbo"
        )

        return heatmap_on_image
