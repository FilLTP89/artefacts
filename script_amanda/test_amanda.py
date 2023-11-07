import argparse, os 
from test import test_single_acquistion, test_metrics



def parse_args():
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument("--generate", default= True)
    argparser.add_argument("--dicom", default=False, type=bool, help="Use dicom data")
    argparser.add_argument("--low", default=False, type=bool, help="Low metal")
    argparser.add_argument("--acquisition_number", default=1, type = int)      
    args = argparser.parse_args()
    return args  


def main(args):
    dicom = args.dicom
    low =  args.low
    acquisition_number = args.acquisition_number

    if args.generate:
        test_single_acquistion(
            dicom = dicom,
            acquisition_number= acquisition_number,
            low = low
        )
    else :
        test_metrics(
            dicom= dicom,
            low = low
        )


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)