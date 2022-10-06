import tensorflow as tf

from . import gatys_2015
from . import johnson_2016


class ReflectionPadding2D(tf.keras.layers.Layer):

    def __init__(self, kernel_size, strides, **kwargs):
        """
        Reflection padding layer.

        Calculate padding in the same way as is done for 'same' padding, based
        on the provided `kernel_size`, `strides`, and input tensor dimensions.

        Args:
            kernel_size:
                An integer.
            strides:
                An integer.

        Input shape:
            A tensor of shape `[B,H,W,C]`.
        """
        super().__init__(**kwargs)

        self.kernel_size = kernel_size
        self.strides = strides

    def call(self, inputs):
        B, H, W, C = tf.unstack(tf.shape(inputs))

        if H % self.strides == 0:
            pad_h = self.kernel_size - self.strides
        else:
            pad_h = self.kernel_size - (H % self.strides)

        if W % self.strides == 0:
            pad_w = self.kernel_size - self.strides
        else:
            pad_w = self.kernel_size - (W % self.strides)

        pad_t = pad_h // 2
        pad_b = pad_h - pad_t
        pad_l = pad_w // 2
        pad_r = pad_w - pad_l

        return tf.pad(
            inputs,
            [[0,0], [pad_t,pad_b], [pad_l,pad_r], [0,0]],
            mode='reflect',
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'strides': self.strides,
        })
        return config


class ConditionalInstanceNormalization(tf.keras.layers.Layer):

    def __init__(self, epsilon=1e-5, **kwargs):
        """
        Conditional Instance Normalization layer.

        Args:
            epsilon:
                    Small float value added to the variance to avoid dividing
                    by zero.

        Input shape:
            A 2-tuple (x,v) of the feature tensor x of shape `[B,H,W,C]` to
            normalize and the style vector v of shape `[B,N]` to use.

        Output shape:
            The normalized feature tensor.

        References:
            * Dumoulin, Shlens, Kudlur.  "A Learned Representation for
            Artistic Style."  ICLR, 2017.
        """
        super().__init__(**kwargs)

        self.epsilon = epsilon

    def build(self, input_shape):

        if not isinstance(input_shape, tuple):
            raise ValueError("Expected tuple input")

        if input_shape[0].rank != 4 or input_shape[1].rank != 2:
            raise ValueError("Expected tuple of tensors of ranks (4,2).")

        C = input_shape[0][-1]                  # number of feature channels
        N = input_shape[1][-1]                  # dimension of style vector

        self.gamma = self.add_weight(name='gamma', shape=[1,1,C,N],
            initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=[1,1,C,N],
            initializer='zeros', trainable=True)

    def call(self, inputs):
        x, v = inputs                                       # [B,H,W,C], [B,N]

        mu, var = tf.nn.moments(x, axes=[1,2], keepdims=True)   # [B,1,1,C]

        x_norm = (x - mu) / tf.sqrt(var + self.epsilon)         # [B,H,W,C]

        v_gamma = tf.tensordot(v, self.gamma, [1,3])            # [B,1,1,C]
        v_beta = tf.tensordot(v, self.beta, [1,3])              # [B,1,1,C]

        return v_gamma * x_norm + v_beta                        # [B,H,W,C]

    def get_config(self):
        config = super().get_config()
        config.update({
            'epsilon':  self.epsilon,
        })
        return config


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
        norm    ConditionalInstanceNormalization
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
        self.norm = ConditionalInstanceNormalization(name='norm')
        self.act = tf.keras.layers.Activation(activation, name="act")

    def call(self, inputs):
        x, v = inputs
        x = self.rpad(x)
        x = self.conv(x)
        x = self.norm((x,v))
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


class UpSamplingBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        filters,
        kernel_size,
        factors,
        activation='relu',
        **kwargs):
        """
        Up-Sampling Block.

        A up-sampling block consists of the following five layers:

        ```text
        up      UpSampling
        rpad    ReflectionPadding
        conv    Convolution
        norm    ConditionalInstanceNormalization
        act     Activation
        ```

        Args:
            filters:
                An integer.  The number of filters of the convolution.
            kernel_size:
                An integer.  The convolution layer's kernel size.
            factors:
                An integer.  The factor(s) by which to upsample the layer's
                spatial dimensions.
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
            isinstance(factors, int)
        ):
            raise TypeError(
                "Parameters 'filters', 'kernel_size', and 'factors' must be"
                "of type int."
            )

        self.filters = filters
        self.kernel_size = kernel_size
        self.factors = factors
        self.activation = activation

        if factors != 1:
            self.up = tf.keras.layers.UpSampling2D(factors)
        else:
            self.up = None

        self.rpad = ReflectionPadding2D(kernel_size, 1)
        self.conv = tf.keras.layers.Conv2D(
            filters, kernel_size, 1,
            padding='valid', use_bias=False, name='conv',
        )
        self.norm = ConditionalInstanceNormalization(name='norm')
        self.act = tf.keras.layers.Activation(activation, name="act")

    def call(self, inputs):
        x, v = inputs

        if self.up is not None:
            x = self.up(x)

        x = self.rpad(x)
        x = self.conv(x)
        x = self.norm((x,v))

        return self.act(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'factors': self.factors,
            'activation': self.activation,
        })
        return config


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, filters=128, **kwargs):
        """
        Residual Block.

        The residual block consists of the following layers:

        ```text
        rpad1   ReflectionPadding
        conv1   Convolution (filters=128, size 3x3, stride 1)
        norm1   ConditionalInstanceNormalization
        act1    ReLU Activation
        rpad2   ReflectionPadding
        conv2   Convolution (filters=128, size 3x3, stride 1)
        norm2   ConditionalInstanceNormalization
        + res   Residual
        ```

        Args:
            filters:
                An integer.  The number of filters of the convolution.

        Call args:
            A tensor of rank 4.

        Returns:
            A tensor of rank 4.
        """
        super().__init__(**kwargs)

        self.filters = filters

        self.rpad1 = ReflectionPadding2D(3, 1)
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, 1, use_bias=False,
                                            name='conv1')
        self.norm1 = ConditionalInstanceNormalization(name='norm1')
        self.act1 = tf.keras.layers.Activation('relu', name='relu')

        self.rpad2 = ReflectionPadding2D(3, 1)
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, 1, use_bias=False,
                                            name='conv2')
        self.norm2 = ConditionalInstanceNormalization(name='norm2')

    def call(self, inputs):
        x, v = inputs

        x = self.rpad1(x)
        x = self.conv1(x)
        x = self.norm1((x,v))
        x = self.act1(x)

        x = self.rpad2(x)
        x = self.conv2(x)
        x = self.norm2((x,v))

        return x + inputs[0]

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
        })
        return config


class MultiStyleLoss:

    def __init__(
        self,
        style_images,
        feature_model='vgg16',
        feature_layers='johnson2016-style',
    ):
        """
        Vectorized multi-style version of Gatys style loss.
        """

        if isinstance(feature_model, str):
            feature_model = gatys_2015.FeatureModel(
                feature_model, feature_layers,
            )

        targets = []
        for features in zip(*tuple(feature_model(image) for image in style_images)):
            gram_features = tuple(
                gatys_2015.avg_gram_tensor(feature)
                for feature in features
            )
            targets.append(gram_features)
        targets = tuple( tf.concat(target, 0) for target in targets )

        self.feature_model = feature_model
        self.targets = targets

    def __call__(self, pastiche_image, style_index):

        targets = tuple(
            tf.gather(target, style_index) for target in self.targets
        )
        features = tuple(
            gatys_2015.avg_gram_tensor(feature)
            for feature in self.feature_model(pastiche_image)
        )

        layer_losses = [
            tf.reduce_mean(tf.square(target-feature), axis=[1,2])
            for target, feature in zip(targets,features)
        ]

        return tf.add_n(layer_losses)


class StyleTransferModel(tf.keras.Model):

    def __init__(
        self,
        style_images,
        filters=(32, 64, 128),
        content_weight=1.0,
        style_weight=1e-4,
        **kwargs):
        """
        Style Transfer Model.

        Dumoulin's style transfer model uses an encoder-decoder architecture
        with residual blocks as bottleneck to facilitate learning of the
        identity transformation.  It is a further development of Johnson's
        style transfer model.  It is fully convolutional and thus can be
        applied to arbitrary size input images.  In the overview below, the
        output size is listed solely to illustrate the bottleneck structure
        and is based on an input image size of 256x256x3.

        ```text
        (Block /) Layer         Description                        Output Size
        ----------------------------------------------------------------------
        input                   Input Layer                        256x256x3

        preprocess              Pre-Processing                     256x256x3

        down_block_1 / conv     Convolution (32, 9x9, stride 1)
                     / norm     ConditionalInstanceNormalization
                     / act      Activation (ReLU)                  256x256x32

        down_block_2 / conv     Convolution (64, 3x3, stride 2)
                     / norm     ConditionalInstanceNormalization
                     / act      Activation (ReLU)                  128x128x64

        down_block_3 / conv     Convolution (128, 3x3, stride 2)
                     / norm     ConditionalInstanceNormalization
                     / act      Activation (ReLU)                  64x64x128

        res_block_1..5 / conv1  Convolution (128, 3x3, stride 1)
                       / norm1  ConditionalInstanceNormalization
                       / relu1  Activation (ReLU)
                       / conv2  Convolution (128, 3x3, stride 1)
                       / norm2  ConditionalInstanceNormalization
                       + crop   Residual = Cropping (2x2)          64x64x128

        up_block_1 / conv       Convolution (64, 3x3, stride 1/2)
                   / norm       ConditionalInstanceNormalization
                   / act        Activation (ReLU)                  128x128x64

        up_block_2 / conv       Convolution (32, 3x3, stride 1/2)
                   / norm       ConditionalInstanceNormalization
                   / act        Activation (ReLU)                  256x256x32

        up_block_3 / conv       Convolution (3, 9x9, stride 1)
                   / norm       ConditionalInstanceNormalization
                   / act        Activation (Sigmoid)               256x256x3

        rescale                 Rescaling (factor 255.0)           256x256x3
        ```

        Remarks:
            * The preprocessing layer normalizes with respect to the ImageNet
              mean RGB values.
            * All convolutions use 'same' amount of reflection padding.

        Args:
            style_image:
                A 4-D image tensor of shape `[1,H,W,3]` representing the style
                image.
            filters:
                A tuple/list of 3 integers.  These indicate the number of
                filters to use in the convolutional and residual blocks.  The
                default is `(32,64,128)`.
            content_weight:
                A float32 value.  The weight of the content loss.
            style_weight:
                A float32 value.  The weight of the style loss.

        Call args:
            A content image.

        Returns:
            The stylized (pastiche) image.

        References:
            * Dumoulin, Shlens, Kudlur.  "A Learned Representation for
              Artistic Style."  ICLR, 2017.
        """
        super().__init__(**kwargs)

        self.style_images = style_images
        self.filters = filters
        self.content_weight = content_weight
        self.style_weight = style_weight

        # Create model layers.
        f1, f2, f3 = filters

        self.prep = johnson_2016.PreProcessing(name='preprocess')
    
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
        #
        # Dumoulin et al use the same feature layers as Johnson et al, but
        # their implementation of vgg16 uses average pooling instead of the
        # default max pooling.  The keras implementation I use doesn't
        # support this, so I should replace it with my own eventually.
        self.content_loss_fn = gatys_2015.ContentLoss(
            feature_model='vgg16', feature_layers='johnson2016-content'
        )
        self.style_loss_fn = MultiStyleLoss(
            style_images,
            feature_model='vgg16', feature_layers='johnson2016-style'
        )

    def call(self, inputs, **kwargs):
        x, v = inputs

        x = self.prep(x)

        x = self.down_block_1((x,v))
        x = self.down_block_2((x,v))
        x = self.down_block_3((x,v))

        for res_block in self.res_blocks:
            x = res_block((x,v))

        x = self.up_block_1((x,v))
        x = self.up_block_2((x,v))
        x = self.up_block_3((x,v))

        return self.rescale(x)

    def train_step(self, data):
        content_image, style_index = data

        n_styles = len(self.style_images)
        style_vector = tf.expand_dims(tf.one_hot(style_index, n_styles), 0)

        with tf.GradientTape() as tape:
            pastiche_image = self((content_image,style_vector), training=True)

            total_loss = (
                self.content_weight * self.content_loss_fn(
                    content_image, None, pastiche_image
                ) +
                self.style_weight * self.style_loss_fn(
                    pastiche_image, style_index
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
            'style_images': [ img.numpy() for img in self.style_images ],
            'filters': self.filters,
            'content_weight': self.content_weight,
            'style_weight': self.style_weight,
        })
        return config

    @classmethod
    def from_config(cls, config):
        style_images = [
            tf.constant(img) for img in config.pop('style_images')
        ]
        return cls(style_images, **config)

    @classmethod
    def from_saved(cls, filepath):
        return tf.keras.models.load_model(
            filepath,
            custom_objects={
                'StyleTransferModel': StyleTransferModel,
            },
        )
