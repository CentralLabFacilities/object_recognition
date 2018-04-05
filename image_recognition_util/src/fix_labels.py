import sys
import os

def fix_label(file, label):

    with open(file) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        content = content[0].split(' ')
    new_content = [label, content[1], content[2], content[3], content[4]]

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
    print 'This script will search for ALL *.txt files recursively in the given path \033[1m' + path + '.\033[0m'
    print 'For every *.txt file found, it will fix the label according to the classNames.txt.\n'

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
        label_txt_path = path+className+"/labels"
        print("class: ",className," with id: ",i)
        for dirname, dirnames, filenames in os.walk(label_txt_path):
            for filename in filenames:
                f = dirname + '/' + filename
                if 'labels' in f and '.txt' in f:
                    print f
                    fix_label(f,i)

    print '\033[1m\033[92mDone!\033[0m'
