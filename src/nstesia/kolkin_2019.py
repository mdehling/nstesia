from math import log2

import tensorflow as tf


def relaxed_earth_movers_distance(C):
    """
    Relaxed Earth Movers Distance.

    Args:
        C:  cost tensor of shape [N,M] or [B,N,M]

    Returns:
        distance tensor of shape [] or [B]
    """
    return tf.maximum(
        tf.reduce_mean(tf.reduce_min(C, axis=-1), axis=-1),
        tf.reduce_mean(tf.reduce_min(C, axis=-2), axis=-1),
    )


def cosine_distance_matrix(A, B):
    """
    Cosine Distance Matrix.

    Args:
        A:  feature tensor of shape [N,C] or [B,N,C]
        B:  feature tensor of shape [M,C] or [B,M,C]

    Returns:
        cosine distance tensor of shape [N,M] or [B,N,M]
    """
    return 1 - tf.math.divide_no_nan(
        tf.matmul(A, B, transpose_b=True),
        tf.matmul(
            tf.norm(A, axis=-1, keepdims=True),
            tf.norm(B, axis=-1, keepdims=True),
            transpose_b=True
        )
    )


def euclidean_distance_matrix(A, B):
    """
    Euclidean Distance Matrix.

    Args:
        A:  feature tensor of shape [N,C] or [B,N,C]
        B:  feature tensor of shape [M,C] or [B,M,C]

    Returns:
        euclidean distance tensor of shape [N,M] or [B,N,M]
    """
    return tf.norm(
        tf.expand_dims(A, axis=-2) - tf.expand_dims(B, axis=-3),
        axis=-1
    )


def content_loss(A, B, epsilon=1e-7):
    """
    Content Loss.

    Args:
        A:  feature tensor of shape [N,C] or [B,N,C]
        B:  feature tensor of shape [M,C] or [B,M,C]

    Returns:
        cost tensor of shape [] or [B]
    """
    dAA = cosine_distance_matrix(A,A)                               # [B,N,M]
    dBB = cosine_distance_matrix(B,B)                               # [B,N,M]

    return tf.reduce_mean(
        tf.abs(
            tf.math.divide_no_nan(
                dAA, tf.reduce_sum(dAA, axis=-2, keepdims=True)
            ) - tf.math.divide_no_nan(
                dBB, tf.reduce_sum(dBB, axis=-2, keepdims=True)
            )
        ),
        axis=[-2,-1]
    )


def moments(A):
    """
    Moments.

    Args:
        A:  feature tensor of shape [N,C] or [B,N,C]

    Returns:
        mean_A:
            tensor of shape [1,C] or [B,1,C] representing the mean
        cov_AA:
            tensor of shape [C,C] or [B,C,C] representing the covariance
    """
    N = A.shape[-2]
    mean_A = tf.reduce_mean(A, axis=-2, keepdims=True)
    cov_AA = (
        tf.matmul(A, A, transpose_a=True) / tf.cast(N, tf.float32)
        - tf.matmul(mean_A, mean_A, transpose_a=True)
    )
    return mean_A, cov_AA


def moment_matching_loss(A, B):
    """
    Moment Matching Loss.
    """
    mean_A, cov_AA = moments(A)
    mean_B, cov_BB = moments(B)

    return (
        tf.reduce_mean(tf.abs(mean_A - mean_B), axis=[-2,-1])
        + tf.reduce_mean(tf.abs(cov_AA - cov_BB), axis=[-2,-1])
    )


class FeatureModel(tf.keras.Model):

    def __init__(
        self,
        feature_norm=255.0,
        name='feature_model',
        **kwargs,
    ):
        """
        Feature Model.
        """
        super().__init__(name=name, **kwargs)

        self.feature_norm = feature_norm

        vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False)
        vgg16.trainable = False

        # In the original article, all except layers 9, 10, 12, and 13 are used.
        feature_layers = [
            'block1_conv1',     #  1
            'block1_conv2',     #  2
            'block2_conv1',     #  3
            'block2_conv2',     #  4
            'block3_conv1',     #  5
            'block3_conv2',     #  6
            'block3_conv3',     #  7
            'block4_conv1',     #  8
            # 'block4_conv2',     #  9
            # 'block4_conv3',     # 10
            'block5_conv1',     # 11
            # 'block5_conv2',     # 12
            # 'block5_conv3',     # 13
        ]

        inputs = vgg16.inputs
        outputs = tuple(
            vgg16.get_layer(layer).output for layer in feature_layers
        )
        self.base_model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name='base_model'
        )
        self.preprocess_fn = tf.keras.applications.vgg16.preprocess_input

    def call(self, image, **kwargs):
        _, H, W, _ = tf.unstack(tf.shape(image))

        image = self.preprocess_fn(image)
        feature_tuple = self.base_model(image)

        features = tf.concat([ image ] +
            [ tf.image.resize(feature, [H,W]) for feature in feature_tuple ],
            axis=-1,
        )

        return features / self.feature_norm


RGB2YCbCr = tf.constant(
    [[ 0.2126, -0.1146,  0.5000]
    ,[ 0.7152, -0.3854, -0.4542]
    ,[ 0.0722,  0.5000, -0.0458]],
    dtype=tf.float32
)


def get_total_loss_fn(
    content_image,
    style_image,
    alpha=16.0,
    sample_grid=(32,32),
    feature_norm=255.0,
):
    feature_model = FeatureModel(feature_norm=feature_norm)

    style_features = feature_model(style_image)
    B, H_s, W_s, C = tf.unstack(tf.shape(style_features))
    style_features = tf.reshape(style_features, [B,H_s*W_s,C])
    style_image = tf.reshape(style_image, [B,H_s*W_s,3])

    content_features = feature_model(content_image)
    B, H_c, W_c, C = tf.unstack(tf.shape(content_features))
    content_features = tf.reshape(content_features, [B,H_c*W_c,C])
    content_image = tf.reshape(content_image, [B,H_c*W_c,3])

    nh, nw = sample_grid
    h0 = tf.random.uniform([], 0, H_c//nh, dtype=H_c.dtype)
    w0 = tf.random.uniform([], 0, W_c//nw, dtype=W_c.dtype)
    grid_indices = tf.reshape(
        tf.expand_dims(tf.range(h0, H_c, H_c//nh), 1) * W_c
        + tf.expand_dims(tf.range(w0, W_c, W_c//nw), 0),
        [nh*nw]
    )

    content_feature_sample = tf.gather(content_features, grid_indices, axis=1)
    # content_image_sample = tf.gather(content_image, grid_indices, axis=1)

    def total_loss_fn(pastiche_image):
        pastiche_features = feature_model(pastiche_image)
        # B, H_p, W_p, C = tf.unstack(tf.shape(pastiche_features))
        pastiche_features = tf.reshape(pastiche_features, [B,H_c*W_c,C])
        pastiche_image = tf.reshape(pastiche_image, [B,H_c*W_c,3])

        pastiche_feature_sample = \
            tf.gather(pastiche_features, grid_indices, axis=1)
        pastiche_image_sample = \
            tf.gather(pastiche_image, grid_indices, axis=1)

        random_indices = tf.random.uniform([nh*nw], 0, H_s*W_s, dtype=tf.int32)

        style_feature_sample = \
            tf.gather(style_features, random_indices, axis=1)
        style_image_sample = tf.gather(style_image, random_indices, axis=1)

        lc = content_loss(pastiche_feature_sample, content_feature_sample)

        lr = relaxed_earth_movers_distance(
            cosine_distance_matrix(
                pastiche_feature_sample,
                style_feature_sample
            )
        )

        lm = moment_matching_loss(
            pastiche_feature_sample,
            style_feature_sample
        )

        pastiche_image_sample = pastiche_image_sample @ RGB2YCbCr
        style_image_sample = style_image_sample @ RGB2YCbCr

        lp = relaxed_earth_movers_distance(
            euclidean_distance_matrix(
                pastiche_image_sample,
                style_image_sample
            ) / 255.0
            + cosine_distance_matrix(
                pastiche_image_sample,
                style_image_sample
            )
        )

        total_loss = alpha*lc + lr + lm + lp/alpha
        total_loss /= 2 + alpha + 1/alpha

        return total_loss

    return total_loss_fn


def laplace_pyramid(image, depth=5):
    _, H, W, _ = tf.unstack(tf.shape(image))

    G = [ tf.image.resize(image, [H//(2**i),W//(2**i)])
          for i in range(depth+1) ]

    L = [ G[i] - tf.image.resize(G[i+1], [H//(2**i),W//(2**i)])
          for i in range(depth) ] + [ G[-1] ]

    return L


def fold_laplace_pyramid(L):
    _, H, W, _ = tf.unstack(tf.shape(L[0]))
    depth = len(L) - 1

    image = L[-1]
    for i in range(1,depth+1):
        h, w = H//(2**(depth-i)), W//(2**(depth-i))
        image = tf.image.resize(image, [h,w]) + L[depth-i]

    return image


@tf.function
def compute_gradients(pastiche_pyramid, loss_fn):
    with tf.GradientTape() as tape:
        pastiche_image = fold_laplace_pyramid(pastiche_pyramid)
        loss = loss_fn(pastiche_image)
    return loss, tape.gradient(loss, pastiche_pyramid)


def generate_step_pastiche_image(
    content_image,
    style_image,
    initial_pastiche_image,
    alpha=1.0,
    iterations=600,
    learning_rate=3e-1,
    sample_grid=(32,32),
    feature_norm=255.0,
):
    loss_fn = get_total_loss_fn(
        content_image,
        style_image,
        alpha=alpha,
        sample_grid=sample_grid,
        feature_norm=feature_norm,
    )
    optimizer = tf.keras.optimizers.RMSprop(learning_rate)

    pastiche_pyramid = [
        tf.Variable(img)
        for img in laplace_pyramid(initial_pastiche_image)
    ]

    for i in range(iterations):
        _, gradients = compute_gradients(pastiche_pyramid, loss_fn)
        optimizer.apply_gradients(zip(gradients,pastiche_pyramid))

    pastiche_image = fold_laplace_pyramid(pastiche_pyramid)
    pastiche_image = tf.clip_by_value(pastiche_image, 0.0, 255.0)

    return pastiche_image


def generate_pastiche_image(
    content_image,
    style_image,
    content_weight=512.0,
    iterations=600,
    learning_rate=1.0,
    feature_norm=255.0,
):
    content_image = content_image
    style_image = style_image

    _, H_c, W_c, _ = tf.unstack(tf.shape(content_image))
    _, H_s, W_s, _ = tf.unstack(tf.shape(style_image))

    scales = int( log2(min(H_c,W_c)) - log2(32) )
    content_pyramid = laplace_pyramid(content_image, depth=scales)

    for i in range(1,scales+1):
        h_c, w_c = H_c//(2**(scales-i)), W_c//(2**(scales-i))
        h_s, w_s = H_s//(2**(scales-i)), W_s//(2**(scales-i))

        if i == 1:
            pastiche_image = (
                content_pyramid[scales-i]
                + tf.reduce_mean(style_image, axis=[1,2], keepdims=True)
            )
        elif i < scales:
            pastiche_image = (
                content_pyramid[scales-i]
                + tf.image.resize(pastiche_image, [h_c,w_c])
            )
        else:
            pastiche_image = tf.image.resize(pastiche_image, [h_c,w_c])
            learning_rate /= 2
            iterations *= 5

        alpha = content_weight * 2**(scales-i)

        pastiche_image = generate_step_pastiche_image(
            tf.image.resize(content_image, [h_c,w_c]),
            tf.image.resize(style_image, [h_s,w_s]),
            pastiche_image,
            alpha=alpha,
            iterations=iterations,
            learning_rate=learning_rate,
            sample_grid=(32,32),
            feature_norm=feature_norm,
        )

    return pastiche_image
