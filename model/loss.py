import tensorflow as tf
from tensorflow.signal import fft2d
from tensorflow.python.ops.numpy_ops import np_config
import tensorflow.keras.backend as K

np_config.enable_numpy_behavior()  # allow the use of numpy operator such as : @ and T


def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def MSE_loss(y_true, y_pred):
    """
    Mean square error loss function
    """
    return tf.reduce_mean(tf.square(y_true - y_pred))


# Loss for the generator, the input does generator -> feature extractor


def style_loss(y_true_extractor_features, y_pred_extractor_features, lambda_list=[]):
    """
    y_true_extractor # B elements list of (bs,Hi,Wi,Ci)
    y_pred_extractor # B elements list of (bs,Hi,Wi,Ci)
    """

    def gram_matrix(input_tensor):
        # Get the batch size, width, height, and number of channels
        _, width, height, channels = K.int_shape(input_tensor)  # (bs,Hi,Wi,Ci)
        # pool_shape = tf.pack([batch_size, width * height, channels])
        input_tensor = K.reshape(
            input_tensor, [-1, width * height, channels]
        )  # (bs,Hi*Wi,Ci)
        gram = (
            tf.transpose(input_tensor, perm=[0, 2, 1])
            @ input_tensor
            / (channels * width * height)  # (bs,Ci,Ci)
        )
        return gram

    # Use lambda_list as hyperparameter for the weight contribution of the different convolutional block
    loss = 0
    for style, generated in zip(y_true_extractor_features, y_pred_extractor_features):
        # Get the number of channels for this feature map

        # Calculate the mean squared difference between the two Gram matrices for this feature map
        gram_style = gram_matrix(style)
        gram_generated = gram_matrix(generated)
        map_loss = tf.reduce_mean(
            tf.norm((gram_style - gram_generated), ord="fro", axis=(1, 2))
        )  # Frobenius norm
        # Add the map loss to the total loss, weighted by 1/4 * (1/channels)^2
        loss += (
            1 / 4
        ) * map_loss  # Multiply by a constant to make the loss more stable

    return loss


# Loss for the generator, the input does generator -> feature extractor
def content_loss(y_true_extractor_features, y_pred_extractor_features, lambda_list=[]):
    """
    List of shape (nb_conv_block,batch_size,Hi,Wi,Ci)
    Need to be list since Hi,Wi,Ci are different for each block
    """
    loss = 0
    for feature_ytrue, feature_ypred in zip(
        y_true_extractor_features, y_pred_extractor_features
    ):
        loss += tf.reduce_mean(tf.norm(feature_ytrue - feature_ypred, ord=2))
    return loss


# Loss for the generator, the input does generator -> discriminator
def perceptual_loss(y_true_discriminator_features, y_pred_discriminator_features):
    loss = 0
    for feature_ytrue, feature_ypred in zip(
        y_true_discriminator_features, y_pred_discriminator_features
    ):
        loss += tf.reduce_mean(tf.abs(feature_ytrue - feature_ypred))
    return loss


# Loss for the generator the input does generator -> discriminator
def generator_gan_loss(y_pred):
    """
    Also called discriminator loss]
    y_pred is the output of the discriminator # (bs, 1)
    """
    return -tf.reduce_mean(tf.math.log(y_pred), axis=0)[0]


# Discriminator loss
def discriminator_loss(y_true_disc, y_pred_disc):
    """
    y_true are the true label : 1 for real image and 0 for fake image
    y_pred is the output of the discriminator # (bs, 1)
    """
    return -tf.reduce_mean(
        tf.math.log(1 - y_pred_disc) + tf.math.log(y_true_disc), axis=0
    )[0]


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
    y_pred = tf.random.uniform(shape=(32, 1))
    y_true = tf.ones_like(y_pred)
    loss = discriminator_loss(y_true, y_pred)
    print(loss)
