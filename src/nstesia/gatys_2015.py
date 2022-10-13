import tensorflow as tf


standard_layers = {
    'vgg16': {
        'johnson2016-content': [
            'block3_conv3',
        ],
        'johnson2016-style': [
            'block1_conv2',
            'block2_conv2',
            'block3_conv3',
            'block4_conv3',
        ],
    },
    'vgg19': {
        'gatys2015a-style': [
            'block1_conv1',
            'block1_pool',
            'block2_pool',
            'block3_pool',
            'block4_pool',
        ],
        'gatys2015b-content': [
            'block4_conv2',
        ],
        'gatys2015b-style': [
            'block1_conv1',
            'block2_conv1',
            'block3_conv1',
            'block4_conv1',
            'block5_conv1',
        ],
    },
}


def FeatureModel(model, layers, name='feature_model', **kwargs):

    if model == 'vgg16':
        base_model = tf.keras.applications.vgg16.VGG16(
            include_top=False, weights='imagenet'
        )
        preprocess_fn = tf.keras.applications.vgg16.preprocess_input
    elif model == 'vgg19':
        base_model = tf.keras.applications.vgg19.VGG19(
            include_top=False, weights='imagenet'
        )
        preprocess_fn = tf.keras.applications.vgg19.preprocess_input
    else:
        raise ValueError("Model not supported")

    base_model.trainable = False

    if isinstance(layers, str):
        layers = standard_layers[model][layers]

    inputs = base_model.inputs
    outputs = tuple(
        base_model.get_layer(layer).output for layer in layers
    )
    base_model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='base_model'
    )

    inputs = tf.keras.Input([None,None,None], name='input')
    x = tf.keras.layers.Lambda(preprocess_fn, name='preprocess')(inputs)
    outputs = base_model(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)


class ContentLoss:

    def __init__(
        self,
        content_image=None,
        feature_model='vgg19',
        feature_layers='gatys2015b-content',
        ):
        """
        Gatys Content Loss.

        Args:
            content_image: The content image.
            feature_model: The content model.  Either a keras model or a
                string.
            feature_layers: The content layers.  Either a string indicating
                default layers, or a sequence of the names of layers as
                strings.

        Call args:
            content_image: The content image.
            style_image: The style image.
            pastiche_image: The pastiche image.

        Returns:
            The loss.
        """

        if isinstance(feature_model, str):
            feature_model = FeatureModel(feature_model, feature_layers)

        if content_image is not None:
            targets = feature_model(content_image)
        else:
            targets = None

        self.feature_model = feature_model
        self.targets = targets

    def __call__(self, content_image, pastiche_image):
        targets = (
            self.targets or self.feature_model(content_image)
        )
        features = self.feature_model(pastiche_image)

        layer_losses = [
            tf.reduce_mean(tf.square(target-feature), axis=[1,2,3])
            for target, feature in zip(targets,features)
        ]

        return tf.add_n(layer_losses)


def avg_gram_tensor(feature):
    """
    Averaged Gram Tensor.

    Given a 4-D feature map $F$, its Gram tensor $G$ is the 3-D tensor with
    components $G_{brs} = \sum_{ij} F_{bijr} F_{bijs}$.  The averaged Gram
    tensor is simply the Gram tensor divided by its number of summands $WH$.

    Args:
        feature:
            A 4-D tensor of shape `[B,H,W,C]` representing a feature map.

    Returns:
        A 3-D tensor of shape `[B,C,C]` representing the averaged Gram tensor.
    """
    B,H,W,C = tf.unstack(tf.shape(feature))

    gram_tensor = tf.linalg.einsum('bijr,bijs->brs', feature, feature)
    return gram_tensor / tf.cast(H*W, tf.float32)


class StyleLoss:

    def __init__(
        self,
        style_image=None,
        feature_model='vgg19',
        feature_layers='gatys2015b-style',
        ):
        """
        Gatys Style Loss.
        """

        if isinstance(feature_model, str):
            feature_model = FeatureModel(feature_model, feature_layers)

        if style_image is not None:
            targets = tuple(
                avg_gram_tensor(target)
                for target in feature_model(style_image)
            )
        else:
            targets = None

        self.feature_model = feature_model
        self.targets = targets

    def __call__(self, style_image, pastiche_image):
        targets = self.targets or tuple(
            avg_gram_tensor(target)
            for target in self.feature_model(style_image)
        )
        features = tuple(
            avg_gram_tensor(feature)
            for feature in self.feature_model(pastiche_image)
        )

        layer_losses = [
            tf.reduce_mean(tf.square(target-feature), axis=[1,2])
            for target, feature in zip(targets,features)
        ]

        return tf.add_n(layer_losses)


class PasticheGenerator:

    def __init__(
        self,
        content_image,
        style_image,
        pastiche_image=None,
        optimizer=None,
        content_weight=1.0,
        style_weight=1e-4,
        var_weight=1e-6,
        ):
        """
        Pastiche Generator.

        Implements the optimization loop combining the content image and style
        image into a pastiche.

        Args:
            content_image:
                A 4-D image tensor of shape `[B,H,W,3]` representing the
                content image.
            style_image:
                A 4-D image tensor of shape `[B,H,W,3]` representing the style
                image.
            initial_image:
                A 4-D image tensor of the same shape as the `content_image` to
                use as initial pastiche image, or `None` for random
                initialization.
            optimizer:
                A keras optimizer.
            content_weight:
                A float32 value.  The weight for the content loss.
            style_weight:
                A float32 value.  The weight for the style loss.
            var_weight:
                A float32 value.  The weight for the variation loss.

        Call args:
            steps:
                An integer.  The number of steps to run the optimization loop
                for.
            iterations_per_step:
                An integer.  Each step amounts to `iterations_per_step`
                iterations of the optimization loop.

        Returns:
            A 4-D image tensor of the same shape as the content image,
            representing the pastiche image.
        """
        self.content_image = content_image
        self.style_image = style_image
        self.pastiche_image = tf.Variable(pastiche_image)

        self.pastiche_images = [tf.identity(pastiche_image)]

        self.optimizer = tf.keras.optimizer.get(optimizer)

        self.content_weight = content_weight
        self.style_weight = style_weight
        self.var_weight = var_weight

        self.content_loss_fn = ContentLoss(content_image=content_image)
        self.style_loss_fn = StyleLoss(style_image=style_image)
        self.losses = []

    @tf.function
    def compute_gradients(self):
        with tf.GradientTape() as tape:
            total_loss = (
                self.content_weight * self.content_loss_fn(
                    self.content_image, self.pastiche_image
                ) +
                self.style_weight * self.style_loss_fn(
                    self.style_image, self.pastiche_image
                ) +
                self.var_weight * tf.image.total_variation(
                    self.pastiche_image
                )
            )

        return total_loss, tape.gradient(total_loss, self.pastiche_image)

    def __call__(self, steps=1, iterations_per_step=100):
        for n in range(steps):
            for m in range(iterations_per_step):
                loss, grad = self.compute_gradients()

                self.optimizer.apply_gradients([(grad,self.pastiche_image)])
                self.pastiche_image.assign(
                    tf.clip_by_value(self.pastiche_image, 0.0, 255.0)
                )

            self.losses.append(loss)
            self.pastiche_images.append( tf.identity(self.pastiche_image) )

        return self.pastiche_images[-1]
