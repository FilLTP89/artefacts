import tensorflow as tf
from vgg19 import VGG19
from MedGAN import MEDGAN, ConsNet
from PatchGAN import PatchGAN
from loss import (
    style_loss,
    content_loss,
    perceptual_loss,
    generator_gan_loss,
    discriminator_loss,
)


def test():
    input = tf.random.normal((3, 512, 512, 1))
    input2 = tf.random.normal((3, 512, 512, 1))

    consnet = ConsNet(6, (512, 512, 1))
    generator_output = consnet(input)

    vgg19 = VGG19()
    feature_extractor_pred_features, feature_extractor_pred__last_layer = vgg19(
        generator_output
    )
    feature_extrator_true_features, feature_extractor_true_last_layer = vgg19(input2)

    patchgan = PatchGAN()
    discriminato_pred_output, discriminator_pred_last_layer = patchgan(generator_output)
    discriminator_true_output, discriminator_true_last_layer = patchgan(input2)

    style_loss_ = style_loss(
        feature_extractor_pred_features, feature_extrator_true_features
    )
    content_loss_ = content_loss(
        feature_extractor_pred_features, feature_extrator_true_features
    )
    perceptual_loss_ = perceptual_loss(
        discriminato_pred_output, discriminator_true_output
    )
    generator_gan_loss_ = generator_gan_loss(discriminator_pred_last_layer)
    discriminator_loss_ = discriminator_loss(
        discriminator_true_last_layer, discriminator_pred_last_layer
    )
    print("style_loss", style_loss_)
    print("content_loss", content_loss_)
    print("perceptual_loss", perceptual_loss_)
    print("generator_gan_loss", generator_gan_loss_)
    print("discriminator_loss", discriminator_loss_)


def patchgan_test():
    patchgan = PatchGAN()
    """ x = tf.random.normal((3, 512, 512, 1))
    y = tf.random.normal((3, 512, 512, 1))
    y_hat_feature, y_hat_lastlayer = patchgan(x)
    y_feature, y_last_layer = patchgan(y)
    print(perceptual_loss(y_hat_feature, y_feature))
    print(generator_gan_loss(y_hat_lastlayer)) """
    x = tf.random.normal((3, 512, 512, 1))
    y_hat = patchgan(x)
    print(y_hat[0][0].shape)
    print(patchgan.trainable_weights)


def vgg19_test():
    vgg19 = VGG19()
    x = tf.random.normal((3, 512, 512, 1))
    y = tf.random.normal((3, 512, 512, 1))
    y_hat_feature = vgg19(x)
    y_feature = vgg19(y)
    print(style_loss(y_hat_feature, y_feature))
    print(content_loss(y_hat_feature, y_feature))


def consnet_test():
    consnet = ConsNet(6, (512, 512, 1))
    x = tf.random.normal((3, 512, 512, 1))
    y_hat = consnet(x)
    print(y_hat.shape)
    print(consnet.trainable_weights.shape)


def medgann_test():
    medgan = MEDGAN((512, 512, 1))
    x = tf.random.normal((3, 512, 512, 1))
    y = tf.random.normal((3, 512, 512, 1))
    medgan.compile()
    medgan.fit(x, y, epochs=1, batch_size=1)
