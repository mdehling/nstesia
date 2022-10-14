import tensorflow as tf

from .gatys_2015 import ContentLoss, StyleLoss
from .johnson_2016 import PreProcessing
from .dumoulin_2017 import ReflectionPadding2D, ResidualBlock, UpSamplingBlock


class DownSamplingBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        filters,
        kernel_size,
        strides,
        activation='relu',
        **kwargs):
        """
        Down-Sampling Block.

        A down-sampling block consists of the following four layers:

        ```text
        rpad    ReflectionPadding
        conv    Convolution
        norm    BatchNormalization
        act     Activation
        ```

        Args:
            filters:
                An integer.  The number of filters of the convolution.
            kernel_size:
                An integer.  The convolution layer's kernel size.
            strides:
                An integer.  The convolution layer's strides.
            activation:
                A string indicating the type of activation to use, e.g.,
                `'relu'` or `'sigmoid'`.

        Call args:
            A tensor of rank 4.

        Returns:
            A tensor of rank 4.
        """
        super().__init__(**kwargs)

        if not (
            isinstance(filters, int) and
            isinstance(kernel_size, int) and
            isinstance(strides, int)
        ):
            raise TypeError(
                "Parameters 'filters', 'kernel_size', and 'strides' "
                "must be integers."
            )

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation

        self.rpad = ReflectionPadding2D(kernel_size, strides)
        self.conv = tf.keras.layers.Conv2D(
            filters, kernel_size, strides,
            padding='valid', use_bias=False, name='conv',
        )
        self.norm = tf.keras.layers.BatchNormalization(name='norm')
        self.act = tf.keras.layers.Activation(activation, name="act")

    def call(self, inputs, training=False):
        x = self.rpad(inputs)
        x = self.conv(x)
        x = self.norm(x, training=training)
        return self.act(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'activation': self.activation,
        })
        return config


class StylePredictionModel(tf.keras.Model):

    def __init__(
        self,
        bottleneck_dim=100,
        **kwargs,
    ):
        """
        Style Prediction Model

        Predict a style vector representing the given style image using a pre-
        trained Inception-V3 network.

        Args:
            bottleneck_dim:
                The bottleneck dimension, i.e., the dimension of the predicted
                style vector.

        Call args:
            A tensor of shape `[B,H,W,3]` representing the style image.

        Returns:
            A style vector of shape `[B,N]` where `N` is the bottleneck
            dimension.
        """
        super().__init__(**kwargs)

        self.bottleneck_dim = bottleneck_dim
        
        base_model = tf.keras.applications.inception_v3.InceptionV3(
            include_top=False, weights='imagenet',
        )
        base_model.trainable = False

        self.prep = tf.keras.applications.inception_v3.preprocess_input
        self.style_model = tf.keras.models.Model(
            inputs=base_model.inputs,
            outputs=base_model.get_layer('mixed7').output,
        )
        self.avg = tf.keras.layers.GlobalAveragePooling2D(
            keepdims=True, name='avg',
        )
        self.bneck = tf.keras.layers.Conv2D(
            bottleneck_dim, [1,1], use_bias=False, name='bneck',
        )

    def call(self, inputs, **kwargs):
        x = self.prep(inputs)                               # [B,H,W,3]
        x = self.style_model(x)                             # [B,14,14,768]
        x = self.avg(x)                                     # [B,1,1,768]
        x = self.bneck(x)                                   # [B,1,1,N]
        return tf.squeeze(x, axis=[1,2])                    # [B,N]

    def get_config(self):
        config = super().get_config()
        config.update({
            'bottleneck_dim': self.bottleneck_dim,
        })
        return config


class StyleTransferModel(tf.keras.Model):

    def __init__(
        self,
        bottleneck_dim=100,
        filters=(32, 64, 128),
        content_weight=1.0,
        style_weight=1e-4,
        **kwargs):
        """
        Style Transfer Model.

        Ghiasi's style transfer model uses essentially the same
        encoder-decoder architecture with residual blocks as bottleneck as was
        introduced by Dumoulin et al.  The only difference in the model itself
        is the use of batch normalization in the encoder part.  The other
        difference is that in Ghiasi's approach a style prediction network is
        employed to learn the mapping from style image to style vector, while
        in Dumoulin's original approach a fixed number of style images was
        used and these images were assigned the standard basis as style
        vectors.

        The transfer model is fully convolutional and thus can be applied to
        arbitrary size input images.  In the overview below, the output size
        is listed solely to illustrate the bottleneck structure and is based
        on an input image size of 256x256x3.

        ```text
        (Block /) Layer         Description                        Output Size
        ----------------------------------------------------------------------
        input                   Input Layer                        256x256x3

        preprocess              Pre-Processing                     256x256x3

        down_block_1 / rpad     ReflectionPadding
                     / conv     Convolution (32, 9x9, stride 1)
                     / norm     BatchNormalization
                     / act      Activation (ReLU)                  256x256x32

        down_block_2 / rpad     ReflectionPadding
                     / conv     Convolution (64, 3x3, stride 2)
                     / norm     BatchNormalization
                     / act      Activation (ReLU)                  128x128x64

        down_block_3 / rpad     ReflectionPadding
                     / conv     Convolution (128, 3x3, stride 2)
                     / norm     BatchNormalization
                     / act      Activation (ReLU)                  64x64x128

        res_block_1..5 / rpad1  ReflectionPadding
                       / conv1  Convolution (128, 3x3, stride 1)
                       / norm1  ConditionalInstanceNormalization
                       / relu1  Activation (ReLU)
                       / rpad2  ReflectionPadding
                       / conv2  Convolution (128, 3x3, stride 1)
                       / norm2  ConditionalInstanceNormalization
                       +        Residual                           64x64x128

        up_block_1 / up         UpSampling (2x)
                   / rpad       ReflectionPadding
                   / conv       Convolution (64, 3x3, stride 1)
                   / norm       ConditionalInstanceNormalization
                   / act        Activation (ReLU)                  128x128x64

        up_block_2 / up         UpSampling (2x)
                   / rpad       ReflectionPadding
                   / conv       Convolution (32, 3x3, stride 1)
                   / norm       ConditionalInstanceNormalization
                   / act        Activation (ReLU)                  256x256x32

        up_block_3 / rpad       ReflectionPadding
                   / conv       Convolution (3, 9x9, stride 1)
                   / norm       ConditionalInstanceNormalization
                   / act        Activation (Sigmoid)               256x256x3

        rescale                 Rescaling (factor 255.0)           256x256x3
        ```

        Remarks:
            * The preprocessing layer normalizes with respect to the ImageNet
              mean RGB values.
            * All convolutions use 'same' amount of reflection padding.

        Args:
            bottleneck_dim:
                An integer.  The dimension of the style vector.
            filters:
                A tuple/list of 3 integers.  These indicate the number of
                filters to use in the convolutional and residual blocks.  The
                default is `(32,64,128)`.
            content_weight:
                A float32 value.  The weight of the content loss.
            style_weight:
                A float32 value.  The weight of the style loss.

        Call args:
            A tuple (x_c,x_s) consisting of a content image and a style image.

        Returns:
            The stylized (pastiche) image.

        References:
            * Ghiasi, Lee, Kudlur, Dumoulin, Shlens.  "Exploring the Structure
              of a Real-Time, Arbitrary Neural Artistic Stylization Network."
              BMVC, 2017.
            * Dumoulin, Shlens, Kudlur.  "A Learned Representation for
              Artistic Style."  ICLR, 2017.
        """
        super().__init__(**kwargs)

        self.bottleneck_dim = bottleneck_dim
        self.filters = filters
        self.content_weight = content_weight
        self.style_weight = style_weight

        # Create style prediction network.
        self.style_predict = StylePredictionModel(
            bottleneck_dim=bottleneck_dim, name='style_predict'
        )

        # Create style transfer model layers.
        f1, f2, f3 = filters

        self.prep = PreProcessing(name='preprocess')
    
        self.down_block_1 = DownSamplingBlock(f1, 9, 1, name='down_block_1')
        self.down_block_2 = DownSamplingBlock(f2, 3, 2, name='down_block_2')
        self.down_block_3 = DownSamplingBlock(f3, 3, 2, name='down_block_3')
    
        self.res_blocks = [
            ResidualBlock(f3, name=f'res_block_{i+1}')
            for i in range(5)
        ]
    
        self.up_block_1 = UpSamplingBlock(f2, 3, 2, name='up_block_1')
        self.up_block_2 = UpSamplingBlock(f1, 3, 2, name='up_block_2')
        self.up_block_3 = UpSamplingBlock(3, 9, 1, activation='sigmoid',
                                          name='up_block_3')

        self.rescale = tf.keras.layers.Rescaling(
            scale=255.0, name='rescale'
        )

        # Create loss functions.
        self.content_loss_fn = ContentLoss(
            feature_model='vgg16', feature_layers='johnson2016-content'
        )
        self.style_loss_fn = StyleLoss(
            feature_model='vgg16', feature_layers='johnson2016-style'
        )

    def call(self, inputs, training=False, **kwargs):
        x_c, x_s = inputs

        v = self.style_predict(x_s)

        y = self.prep(x_c)

        y = self.down_block_1(y, training=training)
        y = self.down_block_2(y, training=training)
        y = self.down_block_3(y, training=training)

        for res_block in self.res_blocks:
            y = res_block((y,v))

        y = self.up_block_1((y,v))
        y = self.up_block_2((y,v))
        y = self.up_block_3((y,v))

        return self.rescale(y)

    def train_step(self, data):
        content_image, style_image = data

        with tf.GradientTape() as tape:
            pastiche_image = self((content_image,style_image), training=True)

            total_loss = (
                self.content_weight * self.content_loss_fn(
                    content_image, pastiche_image
                ) +
                self.style_weight * self.style_loss_fn(
                    style_image, pastiche_image
                )
            )

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))

        return { 'total_loss': total_loss }

    def get_config(self):
        try:
            config = super().get_config()
        except NotImplementedError:
            config = {}

        config.update({
            'bottleneck_dim': self.bottleneck_dim,
            'filters': self.filters,
            'content_weight': self.content_weight,
            'style_weight': self.style_weight,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def from_saved(cls, filepath):
        return tf.keras.models.load_model(
            filepath,
            custom_objects={
                'StyleTransferModel': StyleTransferModel,
            },
        )
