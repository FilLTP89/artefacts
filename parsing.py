import argparse


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="./data/", help="Path of the data folder")
    parser.add_argument("--model", default="Unet", help="Model to train")
    parser.add_argument(
        "--imageHeight", type=int, default=400, help="Height of the input image"
    )
    parser.add_argument(
        "--imageWidth", type=int, default=400, help="Width of the input image"
    )
    parser.add_argument(
        "--imageChannels", type=int, default=1, help="Channels of the input image"
    )
    parser.add_argument("--umin", type=int, default=137, help="Lower grayscale integer")
    parser.add_argument(
        "--umax", type=int, default=52578, help="Upper grayscale integer"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="AE learning rate, default=0.0001"
    )
    parser.add_argument("--cuda", action="store_true", help="enables cuda")
    parser.add_argument("--gpu", type=int, default=2, help="number of GPUs to use")
    # parser.add_argument('--wtdof',nargs='+',default=[3],help='Specify the connection between wdof and tdof (mdof database only)')
    # parser.set_defaults(stack=False,ftune=False,feat=False,plot=True)
    opt = parser.parse_args().__dict__
    return opt
