import sys
import os
import signal
import collections as col
import cv2 as cv

from lxml import etree


object_id = {'0' : 'plate',
             '1' : 'coke',
             '2' : 'sponge',
             '3' : 'cloth',
             '4' : 'ricecrispies',
             '5' : 'cornflakes',
             '6' : 'peanuts',
             '7' : 'biscuits',
             '8' : 'soap',
             '9' : 'pringles'}

width = 640
height = 480

dir_suffix = '_darkflow'

debug = False


def signal_handler(signal, frame):
    print('\n' + '\033[94m' + 'Bye!')
    print  '\033[0m'
    sys.exit(0)

def checkObjectLists(foundClasses):

    if col.Counter(foundClasses) == col.Counter(object_id.values()):
        print 'The found object classes match the ones given in the code!'
        return True

    diff = set(foundClasses).symmetric_difference(set(object_id.values()))

    print '\033[91m' + 'The found classes DON\'T match with the given classes in the code! Make them match to avoid confusion and fucked up data sets!'
    print 'Differences are: ' + str(diff) + '\033[0m'
    print '----------------- found classes:'
    i = 0
    for c in classes:
        i = i + 1
        print c
    print '===> length ' + str(i) # prevents weird len bug
    print '----------------- saved classes: '
    j = 0
    for o in object_id:
        j = j + 1
        print  o + ': ' + object_id[o]
    print '===> length ' + str(j)
    print '-----------------'
    return False


def debugImage(path, classname, p1, p2):

    img = cv.imread(path)
    cv.rectangle(img, p1, p2, (255, 0, 0), 2)
    cv.imshow(classname, img)
    cv.waitKey(1)

    return

def createXML(filename, classname, xmin, xmax, ymin, ymax):
    annotation = etree.Element('annotation')

    fo = etree.Element('folder')
    fo.text = classname + '/images'

    annotation.append(fo)

    f = etree.Element('filename')
    f.text = filename

    annotation.append(f)

    size = etree.Element('size')
    w = etree.Element('width')
    w.text = str(width)
    h = etree.Element('height')
    h.text = str(height)
    d = etree.Element('depth')
    d.text = str(1)

    size.append(w)
    size.append(h)
    size.append(d)

    annotation.append(size)

    seg = etree.Element('segmented')
    seg.text = str(0)

    annotation.append(seg)

    object = etree.Element('object')
    n = etree.Element('name')
    p = etree.Element('pose')
    t = etree.Element('truncated')
    d_1 = etree.Element('difficult')
    bb = etree.Element('bndbox')

    n.text = classname
    p.text = 'center'
    t.text = str(1)
    d_1.text = str(0)

    xmi = etree.Element('xmin')
    ymi = etree.Element('ymin')
    xma = etree.Element('xmax')
    yma = etree.Element('ymax')

    xmi.text = str(xmin)
    yma.text = str(ymax)
    ymi.text = str(ymin)
    xma.text = str(xmax)

    bb.append(xmi)
    bb.append(ymi)
    bb.append(xma)
    bb.append(yma)

    object.append(n)
    object.append(p)
    object.append(t)
    object.append(d_1)
    object.append(bb)

    annotation.append(object)

    return annotation

def saveXML(xml, dir, filename):
    path = dir + '/' + filename

    if(debug):
        print 'Creating file ' + path + ':'

    if not (os.path.exists(dir)):
        print dir + ' does not exist yet, creating it ...'
        os.makedirs(dir)

    with open(path, "w") as file:
        file.write(etree.tostring(xml, pretty_print=True))

def convert(file, inn, out):

    with open(inn + '/' + file) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        info = content[0].split(' ')

    x_center = float(info[1]) * width
    y_center = float(info[2]) * height

    bb_width = float(info[3]) * width
    bb_height = float(info[4]) * height

    xmin = int(x_center - (bb_width / 2))
    xmax = int(x_center + (bb_width / 2))

    ymin = int(y_center - (bb_height / 2))
    ymax = int(y_center + (bb_height / 2))

    filename_jpg = file[:-3] + 'jpg'
    filename_xml = file[:-3] + 'xml'

    classname = object_id[info[0]]

    xml = createXML(filename_jpg, classname, xmin, xmax, ymin, ymax)

    saveXML(xml, out, filename_xml)

    if(debug):
        path = (inn + '/' + filename_jpg).replace('labels', 'images')
        debugImage(path, classname, (xmin, ymin), (xmax, ymax))
        s = etree.tostring(xml, pretty_print=True)
        print s


if __name__ == "__main__":

    # check for correct argument size
    if len(sys.argv) > 3 or len(sys.argv) < 2:
        print '\033[91m' + 'Argument Error!\nUsage: python darknet_to_darkflow.py path_to_dataset [--debug]' + '\033[0m'
        exit(1)
    if not os.path.isdir(sys.argv[1]):
        print '\033[91m' + sys.argv[1] + ' is not a directory!' + '\033[0m'
        exit(1)
    if len(sys.argv) == 3 and sys.argv[2] == '--debug':
        print '\033[92mDebug mode is ON\033[0m'
        debug = True


    print 'Searching dataset in: ' + str(sys.argv[1])

    # add SIGINT handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)

    in_dirs = []
    out_dirs = []
    classes = []

    for dirname, dirnames, filenames in os.walk(sys.argv[1]):
        if 'labels' in dirname and not dir_suffix in dirname:
            in_dirs.append(dirname)
            classes.append((dirname.split('/')[-2]))

    if not checkObjectLists(classes):
        exit(0)

    for dir in in_dirs:
        out_dirs.append(dir.replace('labels', 'labels' + dir_suffix))

    for i, d in enumerate(in_dirs):

        for dirname, dirnames, filenames in os.walk(d):

            for file in filenames:
                convert(file, d, out_dirs[i])
            cv.destroyWindow(classes[i])


