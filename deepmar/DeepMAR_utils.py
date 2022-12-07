# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import argparse
import tensorflow as tf

def ParseOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database',
                        default="./",
                        help='Database main folder')
    parser.add_argument('--trDatabase',
                        default="./",
                        help='Training database subfolder')
    parser.add_argument('--tsDatabase',
                        default="./",
                        help='Testing database subfolder')
    parser.add_argument('--vdDatabase',
                        default="./",
                        help='Validation database subfolder')
    parser.add_argument('--imageHeight', 
                        type=int, 
                        default=400,
                        help='Height of the input image')
    parser.add_argument('--imageWidth', 
                        type=int, 
                        default=400, 
                        help='Width of the input image')
    parser.add_argument('--imageChannels', 
                        type=int, 
                        default=1, 
                        help='Channels of the input image')
    parser.add_argument('--umin', 
                        type=int,                        
                        default=137, 
                        help='Lower grayscale integer')
    parser.add_argument('--umax',
                        type=int,
                        default=52578,
                        help='Upper grayscale integer')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=100, 
                        help='Number of training epochs')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.0001, 
                        help='AE learning rate, default=0.0001')
    parser.add_argument('--cuda', 
                        action='store_true', 
                        help='enables cuda')
    parser.add_argument('--gpu', 
                        type=int, 
                        default=2, 
                        help='number of GPUs to use')
    parser.add_argument('--plot',
                        action='store_true',
                        default=False,
                        help='plot figures')
    parser.add_argument('--preprocess',
                        action='store_true',
                        default=False,
                        help='Preprocess raw images')
    parser.add_argument('--batchSize', 
                        type=int, 
                        default=1, 
                        help='# images in batch')
    parser.add_argument('--nFiltersGenerator', 
                        type=int, 
                        default=64,
                        help='# of generator filters in first conv layer')
    parser.add_argument('--nFiltersDiscriminator',
                        type=int, 
                        default=64,
                        help='# of discriminator filters in first conv layer')
    parser.add_argument('--kh',
                        type=int,
                        default=5,
                        help='along-height kernel size')
    parser.add_argument('--dh',
                        type=int,
                        default=2,
                        help='along-height stride')
    parser.add_argument('--kw',
                        type=int,
                        default=5,
                        help='along-width kernel size')
    parser.add_argument('--dw',
                        type=int,
                        default=2,
                        help='along-width stride')
    parser.add_argument('--stddev',
                        type=float,
                        default=0.02,
                        help='Initializer standard deviation')
    parser.add_argument('--nXchannels',
                        type=int,
                        default=1, 
                        help='# of input image channels')
    parser.add_argument('--output_nc', 
                        type=int,
                        default=1, 
                        help='# of output image channels')
    parser.add_argument('--checkpoint_dir', 
                        default='../checkpoint_dir', 
                        help='models are saved here')
    parser.add_argument("--checkpoint_step",
                        type=int,
                        default=500,
                        help="Checkpoint epochs")
    parser.add_argument('--n',
                        default='2', 
                        help='example number')
    parser.add_argument("--nCritic",
                        type=int,
                        default=1,
                        help='number of discriminator training steps')
    parser.add_argument("--DxTrainType",
                        type=str,
                        default='GAN',
                        help='Train Dx with GAN, WGAN or WGANGP')
    parser.add_argument("--DxLR",
                        type=float,
                        default=0.0001,
                        help='Learning rate for Dx [GAN=0.0002/WGAN=0.001]')
    parser.add_argument("--PenAdvXloss",
                        type=float,
                        default=1.0,
                        help="Penalty coefficient for Adv X loss")
    parser.add_argument("--PenGPXloss",
                        type=float,
                        default=1.0,
                        help="Penalty coefficient for WGAN GP X loss")
    parser.add_argument("--trainVeval",
                        type=str,
                        default="TRAIN",
                        help="Train or Eval")
    # parser.add_argument('--wtdof',nargs='+',default=[3],help='Specify the connection between wdof and tdof (mdof database only)')
    # parser.set_defaults(stack=False,ftune=False,feat=False,plot=True)
    options = parser.parse_args().__dict__
    
    if not os.path.exists(options['checkpoint_dir']):
        os.makedirs(options['checkpoint_dir'])

    options["DeviceName"] = tf.test.gpu_device_name() if not options['cuda'] else "/cpu:0"
    return options