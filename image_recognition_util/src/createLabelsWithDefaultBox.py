import sys
import os

def add_label(file, label):

    new_content = [label, 0.5, 0.5, 1, 1]

    with open(file, "w") as f:
        for e in new_content:
            f.write(str(e) + ' ')

        f.write('\n')

if __name__ == "__main__":

    # check for correct argument size
    if not len(sys.argv) == 2:
        print '\033[91m' + 'Argument Error!\nUsage: python fix_txt.py path_to_dataset' + '\033[0m'
        exit(1)

    # check if argument given is a directory
    if not os.path.isdir(sys.argv[1]):
        print '\033[91m' + sys.argv[1] + ' is not a directory!' + '\033[0m'
        exit(1)

    path = sys.argv[1]

    # some warning
    print 'This script will search for ALL *.jpg files recursively in the given path \033[1m' + path + '.\033[0m'
    print 'For every *.jpg file found, it will add the label file  with default box according to the classNames.txt.\n'

    # make sure, the user knows what will happen
    choice = raw_input('Are you sure, you want to continue? [y/n] ')

    if not choice == 'y' and not choice == 'Y':
        print 'Alright, bye!'
        exit(0)

    # search for textfiles
    class_name_path = path+"/classNames.txt"
    with open(class_name_path) as f:
        classNames = f.readlines()
        classNames = [x.strip() for x in classNames]

    for i in range(0,len(classNames)):
        className = str(classNames[i])
        class_path = path+className
        label_txt_path = class_path+"/labels"
        image_path = class_path+"/images"
        if not os.path.isdir(image_path):
            os.makedirs(image_path)
        if not os.path.isdir(label_txt_path):
            os.makedirs(label_txt_path)

        print("class: ",className," with id: ",i)
        for dirname, dirnames, filenames in os.walk(class_path):
            for filename in filenames:
                if not "images" in filename and ".jpg" in filename:
                    # move image files to image folder
                    cur_image = dirname + '/' + filename
                    new_image = image_path + '/' + filename
                    print ("move image {} to {}".format(cur_image, new_image))
                    os.rename(cur_image, new_image)
                if not "mask" in filename:
                    # add label file
                    f = (label_txt_path + '/' + filename).replace('jpg','txt')
                    print ("add label file: {}".format(f))
                    add_label(f,i)

    print '\033[1m\033[92mDone!\033[0m'
