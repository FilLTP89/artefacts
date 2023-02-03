# TODO : Implement the loss function for the model if needed

import tensorflow as tf
from tensorflow.signal import fft2d


def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def content_loss(y_true, y_pred):

    pass


def MSE_loss(y_true, y_pred):
    """
    Mean square error loss function
    """
    return tf.reduce_mean(tf.square(y_true - y_pred))


def perceptual_loss(y_true_discriminator, y_pred_discriminator):
    loss = 0
    for feature_ytrue, feature_ypred in zip(y_true_discriminator, y_pred_discriminator):
        loss += tf.reduce_mean(tf.abs(feature_ytrue - feature_ypred))
    return loss


def style_loss(style_grams, generated_grams, lambda_list=[]):
    """
    list of shape (nb_conv_block,batch_size,Hi,Wi,Ci)
    Need to be list since Hi,Wi,Ci are different for each block
    """

    def gram_matrix(input_tensor):
        # Get the batch size, width, height, and number of channels
        batch_size, width, height, num_channels = tf.shape(input_tensor)

        # Reshape the input tensor to shape (batch_size * width * height, num_channels)
        features = tf.reshape(input_tensor, (batch_size * width * height, num_channels))

        # Calculate the Gram matrix by multiplying the features with its transpose
        gram = tf.matmul(features, tf.transpose(features)) / (
            num_channels * width * height
        )

        return gram

    # Use lambda_list as hyperparameter for the weight contribution of the different convolutional block
    """ 
    B = len(y_true)
    loss = 0
    for i in range(B):
        loss += (
            lambda_list[i]
            * tf.norm(gram_matrix(y_true[i]) - gram_matrix(y_pred[i])) ** 2
        )  # Frobenius squared norm between the two Gram matrix 
    """

    loss = 0
    for style, generated in zip(style_grams, generated_grams):
        # Get the number of channels for this feature map
        channels = style.shape[-1]

        # Calculate the mean squared difference between the two Gram matrices for this feature map
        map_loss = tf.reduce_mean(
            tf.square(gram_matrix(style) - gram_matrix(generated))
        )  # Frobenius norm

        # Add the map loss to the total loss, weighted by 1/4 * (1/channels)^2
        loss += (
            1 / (4 * channels**2) * map_loss
        )  # Multiply by a constant to make the loss more stable

    return loss


def generator_gan_loss(y_pred):
    """
    Also called discriminator loss]
    y_pred is the output of the discriminator # (bs, 1)
    """
    return -tf.reduce_mean(tf.math.log(y_pred), axis=0)[0]


def discriminator_loss(y_true, y_pred):
    """
    y_true are the true label : 1 for real image and 0 for fake image
    y_pred is the output of the discriminator # (bs, 1)
    """
    return -tf.reduce_mean(tf.math.log(y_true) + tf.math.log(1 - y_pred), axis=0)[0]


def understanding_generator_gan_loss():
    """
    Here what is evaluated is the ability of the generator too fool the discriminator
    We want the discriminator to predict 1 for the fake image
    """
    y_pred = tf.abs(tf.random.normal((3, 1)))
    y_wanted = tf.ones_like(y_pred)
    print(-tf.reduce_mean(tf.math.log(y_pred), axis=0, keepdims=False)[0])
    print(tf.keras.losses.BinaryCrossentropy()(y_wanted, y_pred))
    # The results should be the same


class FocalFrequencyLoss(tf.keras.layers.Layer):
    """The tf.keras.layers.Layer class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False

    Taken from : https://github.com/ZohebAbai/tf-focal-frequency-loss/blob/main/tf_focal_frequency_loss/tf_focal_frequency_loss.py
    """

    def __init__(
        self,
        loss_weight=1.0,
        alpha=1.0,
        patch_factor=1,
        ave_spectrum=False,
        log_matrix=False,
        batch_matrix=False,
    ):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        tf.debugging.assert_equal(
            tf.cast(tf.math.floormod(h, patch_factor), dtype=tf.int32),
            tf.constant(0),
            message="Patch factor should be divisible by image height and width",
        )
        tf.debugging.assert_equal(
            tf.cast(tf.math.floormod(w, patch_factor), dtype=tf.int32),
            tf.constant(0),
            message="Patch factor should be divisible by image height and width",
        )
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(
                    x[
                        :,
                        :,
                        i * patch_h : (i + 1) * patch_h,
                        j * patch_w : (j + 1) * patch_w,
                    ]
                )

        # stack to patch tensor
        y = tf.stack(patch_list, axis=1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        freq = fft2d(tf.cast(y, tf.complex64))
        freq_real = tf.math.real(freq) / freq.shape[-1]
        freq_imag = tf.math.imag(freq) / freq.shape[-1]
        freq = tf.stack([freq_real, freq_imag], axis=-1)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = (
                tf.math.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha
            )

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = tf.math.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / tf.math.reduce_max(matrix_tmp)
            else:
                matrix_tmp = (
                    matrix_tmp
                    / tf.math.reduce_max(tf.reduce_max(matrix_tmp, axis=-1), axis=-1)[
                        :, :, :, None, None
                    ]
                )

            matrix_tmp = tf.where(
                tf.math.is_nan(matrix_tmp), tf.zeros_like(matrix_tmp), matrix_tmp
            )
            weight_matrix = tf.clip_by_value(
                matrix_tmp, clip_value_min=0.0, clip_value_max=1.0
            )

        tf.debugging.assert_greater_equal(
            tf.cast(tf.math.reduce_min(weight_matrix), dtype=tf.int32),
            tf.constant(0),
            message="The values of spectrum weight matrix should be in the range [0, 1]",
        )
        tf.debugging.assert_less_equal(
            tf.cast(tf.math.reduce_max(weight_matrix), dtype=tf.int32),
            tf.constant(1),
            message="The values of spectrum weight matrix should be in the range [0, 1]",
        )

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return tf.math.reduce_mean(loss)

    def call(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (tf.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (tf.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (tf.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred = tf.transpose(
            pred, perm=[0, 2, 3, 1]
        )  # permute since I'm using channels last
        target = tf.transpose(
            target, perm=[0, 2, 3, 1]
        )  # permute since I'm using channels last
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = tf.math.reduce_mean(pred_freq, axis=0, keepdim=True)
            target_freq = tf.math.reduce_mean(target_freq, axis=0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight


if __name__ == "__main__":
    # understanding_generator_gan_loss()
    loss = MSE_loss
    y = tf.random.uniform((32, 256, 256, 1))
    y_pred = tf.random.uniform((32, 256, 256, 1))
    print(FocalFrequencyLoss()(y_pred, y))
