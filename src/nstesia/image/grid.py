__all__ = [
    'grid',
]

import tensorflow as tf


def grid(images, ncols=3, space=32):
    """
    Organize Images on a Grid.

    Args:
        images: A list of 4-D image tensors of shape [1,H,W,3].
        ncols: The number of columns of the grid.
        space: The space in pixels to leave blank between images.

    Returns:
        The grid of images as a 4-D image tensor.
    """
    nrows = (len(images)-1)//ncols + 1
    if nrows == 1:
        ncols = len(images)

    # make sure we have the exact right number of images
    images += [ None ] * (nrows*ncols - len(images))
    images = [ image if image is not None else tf.constant(255.0, shape=(1,0,0,3)) for image in images ]
    # organize them as a grid (nested list)
    images = [ images[i*ncols:(i+1)*ncols] for i in range(nrows) ]

    # find the maximum image height per row and width per column
    shapes = tf.constant([ [image.shape for image in row] for row in images ])
    max_heights = tf.repeat( tf.reduce_max(shapes[:,:,1], axis=1, keepdims=True), ncols, axis=1)
    max_widths = tf.repeat( tf.reduce_max(shapes[:,:,2], axis=0, keepdims=True), nrows, axis=0)
    max_dims = tf.stack([shapes[:,:,0],max_heights, max_widths,shapes[:,:,3]], axis=-1)

    # pad each image to the maximum height and width for its row and column
    paddings = max_dims - shapes + [0,space,space,0]
    paddings = tf.stack([paddings//2, paddings-paddings//2], axis=-1)

    for i in range(nrows):
        for j in range(ncols):
            images[i][j] = tf.pad(images[i][j], paddings[i,j], 'constant', 255.0)
        images[i] = tf.concat(images[i], axis=2)
    grid_image = tf.concat(images, axis=1)

    # shave off outside border (we only want space between images)
    grid_image = grid_image[:,space//2:space//2-space,space//2:space//2-space,:]

    return grid_image
