__all__ = [
    'load_image',
]

import tensorflow as tf


def load_image(filename, target_size=None):
    """
    Load image from file.
    
    Args:
        filename: Filename of the image to load.
        target_size: The target size as a tuple (H,W) or None to keep the
        original size.

    Returns:
        A 4-D tensor of shape [1,H,W,C] with values in the range 0.0..255.0.
    """
    image = tf.keras.utils.load_img(filename)
    image_tensor = tf.keras.utils.img_to_array(image, dtype="float32")
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    if target_size is not None:
        image_tensor = tf.image.resize(image_tensor, target_size)

    return image_tensor
