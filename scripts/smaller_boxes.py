import sys
import os
from fractions import Fraction

width = 640
height = 480

def fix(file):

    with open(file) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        content = content[0].split(' ')

    x1 = float(content[1])
    x2 = float(content[2])
    x3 = float(content[3])
    x4 = float(content[4])

    x1_new = x1 + x3/4
    x2_new = x2 + x4/4
    x3_new = x3/2
    x4_new = x4/2

    new_content = [content[0], x1_new, x2_new, x3_new, x4_new]

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
    print 'For every *.txt file found, it will fix the boundingbox error where width & height were switched.'
    print 'Make sure, that all *.txt files are in the correct format!'
    print 'Otherwise, you will encounter some strange errors or this script will simply fail.\n'

    # make sure, the user knows what will happen
    choice = raw_input('Are you sure, you want to continue? [y/n] ')

    if not choice == 'y' and not choice == 'Y':
        print 'Alright, bye!'
        exit(0)

    # search for textfiles
    for dirname, dirnames, filenames in os.walk(path):

        for filename in filenames:
             f = dirname + '/' + filename

             #
             if 'labels' in f and '.txt' in f:
                fix(f)


    print '\033[1m\033[92mDone!\033[0m'
