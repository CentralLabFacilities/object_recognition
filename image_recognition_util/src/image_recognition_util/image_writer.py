import cv2
import os
import datetime

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)



def write_annotated(dir_path, image, mask, label, cls_id, bbox, test=False):
    """
    Write an image with an annotation to a folder
    :param dir_path: The base directory we are going to write to
    :param image: The OpenCV image
    :param label: The label that is used for creating the sub directory if not exists
    :param verified: Whether we are sure the label is correct
    """

    if dir_path is None:
        return False

    tensor_path = dir_path + "/tensorset"
    darknet_path = dir_path + "/darkset"

    # Check if path exists, otherwise created it
    if not os.path.exists(tensor_path):
        os.makedirs(tensor_path)
    if not os.path.exists(darknet_path):
        os.makedirs(darknet_path)

    # Main directory for files of class <label>
    class_dir = darknet_path + "/" + label
    image_dir_tensor = tensor_path + "/" + label
    if not os.path.exists(image_dir_tensor):
        os.makedirs(image_dir_tensor)

    # Directory for label files
    label_dir = class_dir + "/labels"
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # Directory for image files
    image_dir = class_dir + "/images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # write image in image_dir
    filename = "{}-{}".format(label, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_%f"))
    cv2.imwrite("{}/{}.jpg".format(image_dir, filename), image)
    cv2.imwrite("{}/{}_mask.jpg".format(image_dir, filename), mask)
    #filename = "/home/sarah/object_recogntion/%s.jpg" % (datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_%f"))
    #cv2.imwrite(filename, image)

    #save bounding box for tensorflow train set
    tensorbox = image[bbox[2]:bbox[3], bbox[0]:bbox[1]]
    cv2.imwrite("{}/{}.jpg".format(image_dir_tensor,filename), tensorbox)

    # convert bbox for darknet
    h, w = image.shape[:2] # changed w, h to h, w -- thilo
    bb = convert((w,h), bbox)

    # write converted bbox as label in label_dir
    label_file = open("{}/{}.txt".format(label_dir, filename), 'w')
    label_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    # safe image path to list for training/test set
    if not test:
        file_list = open("{}/train.txt".format(darknet_path),'a')
    else:
        file_list = open("{}/test.txt".format(darknet_path),'a')

    file_list.write("{}/{}.jpg\n".format(image_dir, filename))
    file_list.close()


    return True
