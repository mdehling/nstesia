import tensorflow as tf
import tensorflow_addons as tfa

from . import gatys_2015


class ReflectionPadding2D(tf.keras.layers.Layer):
    """
    Reflection Padding 2D.

    Args:
        padding:
            An integer of a tuple/list of 2 integers.  Indicates vertical and
            horizontal padding.

    Call args:
        A 4-D tensor of shape [B,H,W,C].

    Returns:
        A 4-D tensor.
    """
    def __init__(self, padding=(1,1), **kwargs):
        super().__init__(**kwargs)

        self.padding = padding

        if isinstance(padding, int):
            padding = (padding,padding)

        vpad, hpad = padding
        self.padding_tensor = tf.constant(
            [[0,0], [vpad, vpad], [hpad, hpad], [0,0]]
        )

    def call(self, inputs, **kwargs):
        return tf.pad(inputs, self.padding_tensor, mode='reflect')

    def get_config(self):
        config = super().get_config()
        config.update({
            'padding': self.padding,
        })
        return config


imagenet_mean_rgb = [123.680, 116.779, 103.939]


class PreProcessing(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        """
        Image Pre-Processing.

        Centers the input tensor around the imagenet mean rgb values.

        Call args:
            inputs: A 4-D image tensor of shape `[B,H,W,3]` with values in
            `0.0..255.0`.

        Returns:
            A 4-D image tensor of shape `[B,H,W,3]`.
        """
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return inputs - tf.constant(imagenet_mean_rgb, shape=[1,1,1,3])


class PostProcessing(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        """
        Image Post-Processing.

        Shift the image tensor by imagenet mean and clip to values in range
        `0.0..255.0`.

        Call args:
            inputs: A 4-D image tensor of shape `[B,H,W,3]`.

        Returns:
            A 4-D image tensor of shape `[B,H,W,3]` with values in
            `0.0..255.0`.
        """
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = inputs + tf.constant(imagenet_mean_rgb, shape=[1,1,1,3])
        return tf.clip_by_value(x, 0.0, 255.0)


class ConvolutionalBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        filters=3,
        kernel_size=(3,3),
        strides=(2,2),
        transpose=False,
        normalization='batch',
        activation='relu',
        **kwargs):
        """
        Convolutional Block.

        A convolutional block consists of the following three layers:

        ```text
        conv    Convolution
        norm    Batch/InstanceNormalization (or None)
        act     Activation
        ```

        Args:
            filters:
                An integer.  The number of filters of the convolution.
            kernel_size:
                An integer or a pair of integers.  The convolution layer's
                kernel size.
            strides:
                An integer or pair ofintegers.  The convolution layer's
                strides.
            transpose:
                A boolean.  Whether to use a transposed convolutional layer
                instead of a regular one.
            normalization:
                A string or `None`.  The type of normalization layer to use.
                One of `'batch'` or `'instance'`, or `None`.
            activation:
                A string indicating the type of activation to use, e.g.,
                `'relu'` or `'tanh'`.

        Call args:
            A tensor of rank 4.

        Returns:
            A tensor of rank 4.

        Raises:
            ValueError:
                If `normalization` contains an invalid value.
        """
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.transpose = transpose
        self.normalization = normalization
        self.activation = activation

        if transpose is True:
            self.conv = tf.keras.layers.Conv2DTranspose(
                filters, kernel_size,
                strides=strides, padding='same', name='conv_t'
            )
        else:
            self.conv = tf.keras.layers.Conv2D(
                filters, kernel_size,
                strides=strides, padding='same', name='conv'
            )

        if normalization is None:
            self.norm = None
        elif normalization == 'batch':
            self.norm = tf.keras.layers.BatchNormalization(name='norm')
        elif normalization == 'instance':
            self.norm = tfa.layers.InstanceNormalization(name='norm')
        else:
            raise ValueError('Unknown type of normalization layer.')

        self.act = tf.keras.layers.Activation(activation, name="act")

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        if self.norm is not None:
            x = self.norm(x, training=training)
        return self.act(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'transpose': self.transpose,
            'normalization': self.normalization,
            'activation': self.activation,
        })
        return config


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, filters=128, normalization='batch', **kwargs):
        """
        Residual Block.

        The residual block consists of the following layers:

        ```text
        conv1   Convolution (filters=128, size 3x3, stride 1)
        norm1   Batch/InstanceNormalization
        act1    ReLU Activation
        conv2   Convolution (filters=128, size 3x3, stride 1)
        norm2   Batch/InstanceNormalization
        + crop  Residual = Cropping (2x2)
        ```

        The convolutional layers use 'valid' padding.  Since this reduces the
        size of the tensor, cropping is applied in the residual connection.

        Args:
            filters:
                An integer.  The number of filters of the convolution.
            normalization:
                A string.  The type of normalization layer to use.  One of
                `'batch'` or `'instance'`.

        Call args:
            A tensor of rank 4.

        Returns:
            A tensor of rank 4.

        Raises:
            ValueError:
                If `normalization` contains an invalid value.
        """
        super().__init__(**kwargs)

        self.filters = filters
        self.normalization = normalization

        self.conv1 = tf.keras.layers.Conv2D(filters, (3,3), name='conv1')
        self.relu1 = tf.keras.layers.Activation('relu', name='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters, (3,3), name='conv2')

        if normalization == 'batch':
            self.norm1 = tf.keras.layers.BatchNormalization(name='norm1')
            self.norm2 = tf.keras.layers.BatchNormalization(name='norm2')
        elif normalization == 'instance':
            self.norm1 = tfa.layers.InstanceNormalization(name='norm1')
            self.norm2 = tfa.layers.InstanceNormalization(name='norm2')
        else:
            raise ValueError('Unknown type of normalization layer.')

        self.crop = tf.keras.layers.Cropping2D(2, name='cropping')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.norm1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x, training=training)

        return x + self.crop(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'normalization': self.normalization
        })
        return config


class StyleTransferModel(tf.keras.Model):

    def __init__(
        self,
        style_image,
        normalization='batch',
        filters=(32, 64, 128),
        tanh_factor=150.0,
        content_weight=1.0,
        style_weight=1e-4,
        var_weight=1e-6,
        **kwargs):
        """
        Style Transfer Model.

        Johnson's style transfer model uses an encoder-decoder architecture
        with residual blocks as bottleneck to facilitate learning of the
        identity transformation.  It is fully convolutional and thus can be
        applied to arbitrary size input images.  In the overview below, the
        output size is listed solely to illustrate the bottleneck structure
        and is based on an input image size of 256x256x3.

        ```text
        (Block /) Layer         Description                        Output Size
        ----------------------------------------------------------------------
        input                   Input Layer                        256x256x3

        preprocess              Pre-Processing                     256x256x3
        rpad                    Reflection Padding (40x40)         336x336x3

        conv_block_1 / conv     Convolution (32, 9x9, stride 1)
                     / norm     Batch/InstanceNormalization
                     / act      Activation (ReLU)                  336x336x3

        conv_block_2 / conv     Convolution (64, 3x3, stride 2)
                     / norm     Batch/InstanceNormalization
                     / act      Activation (ReLU)                  168x168x64

        conv_block_3 / conv     Convolution (128, 3x3, stride 2)
                     / norm     Batch/InstanceNormalization
                     / act      Activation (ReLU)                  84x84x128

        res_block_1..5 / conv1  Convolution (128, 3x3, stride 1)
                       / norm1  Batch/InstanceNormalization
                       / relu1  Activation (ReLU)
                       / conv2  Convolution (128, 3x3, stride 1)
                       / norm2  Batch/InstanceNormalization
                       + crop   Residual = Cropping (2x2)          80x80x128..
                                                                   ..64x64x128

        conv_block_4 / conv     Convolution (64, 3x3, stride 1/2)
                     / norm     Batch/InstanceNormalization
                     / act      Activation (ReLU)                  128x128x64

        conv_block_5 / conv     Convolution (32, 3x3, stride 1/2)
                     / norm     Batch/InstanceNormalization
                     / act      Activation (ReLU)                  256x256x32

        conv_block_6 / conv     Convolution (3, 9x9, stride 1)
                     / act      Activation (TanH)                  256x256x3

        rescale                 Rescaling (factor 150.0)           256x256x3
        postprocess             Post-Processing                    256x256x3
        ```

        Remarks:
            * The pre- and postprocessing layers (de)normalize with respect to
              the ImageNet mean RGB values.  Postprocessing also performs
              clipping to the 0.0..255.0 range.
            * The convolutional blocks use 'same' padding, the residual blocks
              use none.  Instead, a layer of reflection padding is applied
              early in the model.  This approach is intended to minimize
              boundary artifacts.
            * Since the convolutions in the residual blocks use no padding,
              their output is 2 pixels smaller on all sides.  Hence, the
              residual connection cannot be an identity map and instead a 2x2
              cropping layer is used.
            * Ulyanov et al suggested using instance normalization instead of
              batch normalization for better results.

        Args:
            style_image:
                A 4-D image tensor of shape `[1,H,W,3]` representing the style
                image.
            normalization:
                A string.  One of `'batch'` or `'instance'` indicating the
                type of normalization layer to use.  Originally, in Johnson et
                al (2016), batch normalization layers were used.  It was shown
                by Ulyanov et al (2016), that replacing these by instance
                normalization layers leads to a significant improvement in
                results.
            filters:
                A tuple/list of 3 integers.  These indicate the number of
                filters to use in the convolutional and residual blocks.  The
                default is `(32,64,128)`.
            tanh_factor:
                A float32 value.  The factor to multiply the final tanh
                activation by before post-processing the result.
            content_weight:
                A float32 value.  The weight of the content loss.
            style_weight:
                A float32 value.  The weight of the style loss.
            var_weight:
                A float32 value.  The weight of the variation loss.

        Call args:
            A content image.

        Returns:
            The stylized (pastiche) image.

        References:
            * Johnson, Alahi, Fei-Fei.  "Perceptual Losses for Real-Time Style
              Transfer and Super-Resolution."  ECCV, 2016.
            * Ulyanov, Vedaldi, Lempitsky.  "Instance Normalization: The
              Missing Ingredient for Fast Stylization."  Arxiv, 2016.
        """
        super().__init__(**kwargs)

        self.style_image = style_image
        self.normalization = normalization
        self.filters = filters
        self.tanh_factor = tanh_factor
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.var_weight = var_weight

        # Create model layers.
        f1, f2, f3 = filters

        self.preprocess = PreProcessing(name='preprocess')
        self.rpad = ReflectionPadding2D(padding=(40,40), name='rpad')
    
        self.conv_block_1 = ConvolutionalBlock(
            filters=f1, kernel_size=(9,9), strides=(1,1),
            normalization=normalization, name='conv_block_1'
        )
        self.conv_block_2 = ConvolutionalBlock(
            filters=f2, normalization=normalization, name='conv_block_2'
        )
        self.conv_block_3 = ConvolutionalBlock(
            filters=f3, normalization=normalization, name='conv_block_3'
        )
    
        self.res_blocks = [
            ResidualBlock(
                filters=f3, normalization=normalization,
                name=f'res_block_{i+1}'
            )
            for i in range(5)
        ]
    
        self.conv_block_4 = ConvolutionalBlock(
            filters=f2, transpose=True,
            normalization=normalization, name='conv_block_4'
        )
        self.conv_block_5 = ConvolutionalBlock(
            filters=f1, transpose=True,
            normalization=normalization, name='conv_block_5'
        )
        self.conv_block_6 = ConvolutionalBlock(
            filters=3, kernel_size=(9,9), strides=(1,1),
            normalization=None, activation='tanh', name='conv_block_6'
        )

        self.rescaling = tf.keras.layers.Rescaling(
            scale=tanh_factor, name='rescale'
        )
        self.postprocess = PostProcessing(name='postprocess')

        # Create loss functions.
        self.content_loss_fn = gatys_2015.ContentLoss(
            feature_model='vgg16', feature_layers='johnson2016-content'
        )
        self.style_loss_fn = gatys_2015.StyleLoss(
            style_image=self.style_image,
            feature_model='vgg16', feature_layers='johnson2016-style'
        )

    def call(self, inputs, training=False):
        x = self.preprocess(inputs)
        x = self.rpad(x)

        x = self.conv_block_1(x, training=training)
        x = self.conv_block_2(x, training=training)
        x = self.conv_block_3(x, training=training)

        for res_block in self.res_blocks:
            x = res_block(x, training=training)

        x = self.conv_block_4(x, training=training)
        x = self.conv_block_5(x, training=training)
        x = self.conv_block_6(x, training=training)

        x = self.rescaling(x)
        return self.postprocess(x)

    def train_step(self, content_image):
        with tf.GradientTape() as tape:
            pastiche_image = self(content_image, training=True)

            total_loss = (
                self.content_weight * self.content_loss_fn(
                    content_image, self.style_image, pastiche_image
                ) +
                self.style_weight * self.style_loss_fn(
                    content_image, self.style_image, pastiche_image
                ) +
                self.var_weight * tf.image.total_variation(
                    pastiche_image
                )
            )

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))

        return { 'total_loss': total_loss }

    def get_config(self):
        config = super().get_config()
        config.update({
            'style_image': self.style_image.numpy(),
            'normalization': self.normalization,
            'filters': self.filters,
            'tanh_factor': self.tanh_factor,
            'content_weight': self.content_weight,
            'style_weight': self.style_weight,
            'var_weight': self.var_weight,
        })
        return config

    @classmethod
    def from_config(cls, config):
        style_image = tf.constant(config.pop('style_image'))
        return cls(style_image, **config)

    @classmethod
    def from_saved(cls, filepath):
        return tf.keras.models.load_model(
            filepath,
            custom_objects={
                'StyleTransferModel': StyleTransferModel,
            },
        )
