from typing import Tuple, Union, List

import numpy as np
from PIL import Image, ImageEnhance, ImageDraw

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

DEF_HEAT_COLORMAP = 'inferno'
GRAYSCALE_CONV = np.array([0.2989, 0.5870, 0.1140])


def fade_image(img: Image.Image, fade_level: float) -> Image.Image:
    """
    Fades the given image by reducing its brightness by a given amount.
    :param Image.Image img: the original image frame.
    :param float fade_level: the fade level in [0,1], where 0 corresponds to the original image, 1 to a black surface.
    :rtype: Image.Image
    :return: the faded image.
    """
    return ImageEnhance.Brightness(img).enhance(1 - fade_level)


def resize_image_canvas(img: Image.Image, size: Tuple[int, int], color: Union[int, str, Tuple[int]] = 0) -> Image.Image:
    """
    Places the given image in a blank canvas of a desired size without resizing the original image itself.
    :param Image.Image img: the original for which to resize the canvas.
    :param (int,int) size: the desired size of the image.
    :param int or str or tuple[int] color: the "blank" color to initialize the new image.
    :rtype: Image.Image
    :return: a new image placed in the blank canvas or the original image, if it's already of the desired size.
    """
    if img.size != size:
        new_img = Image.new(img.mode, size, color)
        new_img.paste(img)
        img = new_img
    return img


def get_max_size(imgs: List[Image.Image]) -> Tuple[int, int]:
    """
    Gets the maximum image size among the given images.
    :param list[Image.Image] imgs: the images from which to get the max size.
    :rtype: (int, int)
    :return: a tuple containing the maximum image size among the given images.
    """
    max_size = [0, 0]
    for img in imgs:
        max_size[0] = max(max_size[0], img.width)
        max_size[1] = max(max_size[1], img.height)
    return max_size[0], max_size[1]


def get_mean_image(imgs: List[Image.Image], canvas_color: Union[int, str, Tuple[int]] = 0) -> Image.Image:
    """
    Gets an image representing the mean of the given images.
    See: https://stackoverflow.com/a/17383621
    :param list[Image.Image] or np.ndarray imgs: the images to be converted.
    :param int or str or tuple[int] canvas_color: the "blank" color to fill in the canvas of out-of-size frames.
    :rtype: Image.Image
    :return: an image representation of the pixel-mean between the given images.
    """
    if isinstance(imgs[0], Image.Image):
        max_size = get_max_size(imgs)
        imgs = np.array(
            [np.array(resize_image_canvas(img, max_size, canvas_color), dtype=np.float) for img in imgs])
    return Image.fromarray(np.array(np.round(imgs.mean(axis=0)), dtype=np.uint8))


def get_variance_heatmap(imgs: List[Image.Image],
                         normalize: bool = True,
                         std_dev: bool = False,
                         color_map: str = DEF_HEAT_COLORMAP,
                         canvas_color: Union[int, str, Tuple[int]] = 0) -> Image.Image:
    """
    Gets the variance of the given images as a heatmap image.
    See: https://stackoverflow.com/a/59537945;
        https://stackoverflow.com/a/17383621
    :param list[Image.Image] or np.ndarray imgs: the images to be converted.
    :param bool normalize: whether to normalize the image values before computing the variance.
    :param str color_map: the name of the matplotlib colormap to produce the heatmap.
    :param bool std_dev: whether to compute standard deviation instead of variance.
    :param int or str or tuple[int] canvas_color: the "blank" color to fill in the canvas of out-of-size frames.
    :rtype: Image.Image
    :return: an image representation of the pixel-variance between the given images.
    """
    from matplotlib import pyplot as plt

    if isinstance(imgs[0], Image.Image):
        max_size = get_max_size(imgs)
        imgs = np.array(
            [np.array(resize_image_canvas(img, max_size, canvas_color), dtype=np.float) for img in imgs])

    # first convert images to grayscale
    imgs = np.dot(imgs[..., :3], GRAYSCALE_CONV)

    # get (normalized) variance
    norm_factor = (2 ** 8) if normalize else 1
    imgs /= norm_factor
    img = ((imgs.std(axis=0) if std_dev else imgs.var(axis=0)) * norm_factor).astype(np.uint8)

    # convert to heatmap
    colormap = plt.get_cmap(color_map)
    img = (colormap(img) * 2 ** 8).astype(np.uint8)[..., :3]

    return Image.fromarray(img)


def overlay_square(img: Image.Image, x: int, y: int, width: int, height: int, color: List[int]) -> Image.Image:
    """
    Draws a semi-transparent colored square over the given image.
    :param Image.Image img: the original image.
    :param int x: the left location relative to the original image where the square should be drawn.
    :param int y: the top location relative to the original image where the square should be drawn.
    :param int width: the width of the rectangle to be drawn.
    :param int height: the height of the rectangle to be drawn.
    :param list[int] color: the color of the square to be drawn, in the RGB (assumes fully opaque) or RGBA format.
    :rtype: Image.Image
    :return: the original image with a square overlaid.
    """
    # make blank, fully transparent image the same size and draw a semi-transparent colored square on it
    alpha = color[3] if len(color) >= 4 else 255
    color = color[:3]
    overlay = Image.new('RGBA', img.size, color + [0])
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(((x, y), (x + width, y + height)), fill=color + [alpha])

    # alpha composite the two images together
    return Image.alpha_composite(img, overlay)
