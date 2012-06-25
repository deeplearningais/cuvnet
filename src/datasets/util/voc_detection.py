#!/usr/bin/python
import sys
import os
from glob import glob
import xml.etree.ElementTree as et


def get_image_props(xml_filename, obj_filter):
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
        if not obj_filter(d):
            continue
        O.append(d)
    return {"objects": O}


def load_metadata(basepath, db, img_filter, obj_filter):
    """
    for a given dataset (val, trainval), load
    meta-data of all contained images.
    """
    dataset_path = os.path.join(basepath,
            'TrainVal/VOCdevkit/VOC2011/ImageSets/Main/*_%s.txt' % db)
    anno_path = os.path.join(basepath,
            'TrainVal/VOCdevkit/VOC2011/Annotations/%s.xml')
    classfiles = glob(dataset_path)
    D = {}
    for cf in classfiles:
        with open(cf, "r") as f:
            for l in f:
                image_name = l.split()[0]
                if image_name in D:
                    continue
                props = get_image_props(anno_path % image_name, obj_filter)
                if not img_filter(props):
                    continue
                D[image_name] = props
    return D


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
          - 4 numbers describing bounding box of object (xmin xmax ymin ymax)
    """
    classes = get_classnames(D)
    images_path = os.path.join(basepath,
            'TrainVal/VOCdevkit/VOC2011/JPEGImages/%s.jpg')
    for name, props in D.items():
        L = []
        L.append(images_path % name)
        L.append(len(props["objects"]))
        if L[-1] == 0:
            continue
        for o in props["objects"]:
            L.append(classes.index(o["name"]))
            L.extend(o["bndbox"])
        f.write(" ".join([str(x) for x in L]))
        f.write("\n")


if __name__ == "__main__":
    basepath = '/home/local/datasets/VOC2011'
    if len(sys.argv) == 2:
        basepath = sys.argv[1]
    #obj_filter = lambda x: x["truncated"] == False
    obj_filter = lambda x: True
    img_filter = lambda x: True
    for dset in ["trainval", "val"]:
        D = load_metadata(basepath, dset, img_filter, obj_filter)
        print dset, sum((len(x["objects"]) for x in D.values()))
        with open("voc_detection_%s.txt" % dset, "w") as f:
            write_for_cpp(f, basepath, D)
