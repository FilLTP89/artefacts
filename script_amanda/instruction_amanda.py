import os
def ask():
    print("What do you want to do?")
    print("1. Generate single acquisition")
    print("2. Test metrics")
    print("3. Exit")
    method = int(input("Enter your choice: "))
    if method == 3:
        print("Exiting...")
        exit()

    print("Do you want to work on dicom or raw data?")
    print("1. Dicom")
    print("2. Raw data")
    data_type = int(input("Enter your choice: "))
    dicom = True if data_type == 1 else False

    print("Low or High amount of metal?")
    print("1. Low")
    print("2. High")
    low_high = int(input("Enter your choice: "))
    low_high = True if low_high == 1 else False

    print("Acquisition number you want to use (0-10):")
    acquisition_number = int(input("Enter your choice: "))
    
    if method == 1:
        """ Generate images for a single acqusition"""
        command = f"sbatch script_amanda/command.sh {True} {dicom} {low_high} {acquisition_number} "
    elif method == 2:
        """Test the metrics"""
        command = f"sbatch script_amanda/command.sh {False} {dicom} {low_high} {acquisition_number} "
    os.system(command)
    


if __name__ == "__main__":
    ask()