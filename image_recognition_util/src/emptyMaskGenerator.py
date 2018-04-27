import sys
import shutil
import os



if __name__ == "__main__":

    # check for correct argument size
    if not len(sys.argv) == 2:
        print '\033[91m' + 'Argument Error!\nUsage: python emptyMaskGenerator.py path_to_dataset' + '\033[0m'
        exit(1)

    # check if argument given is a directory
    if not os.path.isdir(sys.argv[1]):
        print '\033[91m' + sys.argv[1] + ' is not a directory!' + '\033[0m'
        exit(1)

    path = sys.argv[1]
    dummy_mask = path + '/mask.jpg'

    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if ".jpg" in filename and not "mask" in filename:
                f = dirname + '/' + filename
                if "tensorset" in f:
                    continue
                new_mask_path = f.replace(".jpg", "_mask.jpg")
                print("create mask: {}".format(new_mask_path))
                shutil.copy(dummy_mask, new_mask_path)