#!/usr/bin/python
import sys
import os
from glob import glob
import xml.etree.ElementTree as et


def get_image_props(xml_filename):
    """ put the contents of the XML file into a Python data structure """
    dom = et.parse(xml_filename)
    root = dom.getroot()
    objects = root.findall("object")
    O = []
    for o in objects:
        d = {}
        d["name"] = o.find("name").text
        d["truncated"] = int(o.find("truncated").text)
        d["difficult"] = int(o.find("difficult").text)
        d["pose"] = o.find("pose").text
        bb = o.find("bndbox")
        d["bndbox"] = [
                int(bb.find("xmin").text),
                int(bb.find("xmax").text),
                int(bb.find("ymin").text),
                int(bb.find("ymax").text)]
        O.append(d)
    return {"objects": O}


def load_metadata(basepath, db, img_filter, obj_filter):
    """
    for a given dataset (val, trainval), load
    meta-data of all contained images.
    """
    dataset_path = os.path.join(basepath,
            'ImageSets/Main/*_%s.txt' % db)
    anno_path = os.path.join(basepath,
            'Annotations/%s.xml')
    classfiles = glob(dataset_path)
    D = {}
    for cf in classfiles:
        with open(cf, "r") as f:
            for l in f:
                image_name = l.split()[0]
                if image_name in D:
                    continue
                props = get_image_props(anno_path % image_name)
                D[image_name] = props
    all_classes = get_classnames(D)
    D2 = {}
    for image, props in D.iteritems():
        objects = []
        for o in props["objects"]:
            inc, classid = obj_filter(o, all_classes)
            if inc:
                o["classid"] = classid
                objects.append(o)
        props2 = {"objects": objects}
        if img_filter(props2):
            D2[image] = props2
    return D2


def get_classnames(D):
    """ collect all classnames from the dataset """
    class_names = {}
    for name, props in D.items():
        for o in props["objects"]:
            class_names[o["name"]] = 1
    return sorted(class_names.keys())


def write_for_cpp(f, basepath, D):
    """
    write data for processing by c++
    the format is (space separated):
        - full path of image
        - number of objects
        - for each object,
          - class index of object
          - 0/1 whether truncated
          - 4 numbers describing bounding box of object (xmin xmax ymin ymax)
    """
    classes = get_classnames(D)
    images_path = os.path.join(basepath,
            'JPEGImages/%s.jpg')
    for name, props in D.items():
        L = []
        L.append(images_path % name)
        L.append(len(props["objects"]))
        if L[-1] == 0:
            continue
        for o in props["objects"]:
            L.append(o["classid"])
            L.append(o["truncated"])
            L.extend(o["bndbox"])
        f.write(" ".join([str(x) for x in L]))
        f.write("\n")


if __name__ == "__main__":
    import argparse
    import re
    parser = argparse.ArgumentParser(description='Generate an easy-to-read (for C++) aggregation of a VOC database')
    parser.add_argument('-b','--basepath',
                        default='/home/local/backup/VOCdevkit/VOC2007',
                        help='set the base path of the VOC database to be used')
    parser.add_argument('-c','--classes',
                        default=[], nargs='*', help='All classes that should be included')
    args = parser.parse_args()

    year = re.search(r"""(20\d\d)""", args.basepath).group(1)

    def obj_filter(x, all_classes):
        #obj_filter = lambda x: x["truncated"] == False
        if args.classes == []:
            return True, all_classes.index(x["name"])
        elif x["name"] in args.classes:
            return True, args.classes.index(x["name"])
        else:
            return False, 0
    def img_filter(x):
        return len(x["objects"]) > 0

    ds_identifier = "_".join(args.classes)
    if len(ds_identifier):
        ds_identifier = ds_identifier + "_"

    for dset in ["train", "trainval", "val", "test"]:
        D = load_metadata(args.basepath, dset, img_filter, obj_filter)
        print dset, len(D), sum((len(x["objects"]) for x in D.values()))
        dest = os.path.join(args.basepath, "ds_VOC%s_%s%s.txt" % (year, ds_identifier, dset))
        with open(dest, "w") as f:
            write_for_cpp(f, args.basepath, D)
