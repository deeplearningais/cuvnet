import os
from glob import glob
import numpy as np
from random import shuffle, seed
from scipy.io import loadmat
import matplotlib as plt
from scipy.ndimage import zoom
import Image
from joblib import Parallel, delayed
import tarfile
import ImageFile
import cv2

class ImageNetData(object):

    def __init__(self):
        self.meta_path = "/home/local/datasets/ILSVRC2014_devkit/data"
        #self.meta_path = "/home/local/backup/ILSVRC2011/ILSVRC2011_devkit-2.0/data"
        #self.meta_path = "/home/local/backup/ILSVRC2010/devkit-1.0/data"
        self.meta_data = loadmat(os.path.join(self.meta_path, "meta_clsloc.mat"), struct_as_record=False)

        self.synsets = np.squeeze(self.meta_data['synsets'])
        #self.ids = np.squeeze(np.array([x.ILSVRC2010_ID for x in self.synsets])) - 1  # -1 for matlab -> "normal"
        #self.ids = np.squeeze(np.array([x.ILSVRC2011_ID for x in self.synsets])) - 1  # -1 for matlab -> "normal"
        self.ids = np.squeeze(np.array([x.ILSVRC2014_ID for x in self.synsets])) - 1  # -1 for matlab -> "normal"
        idx = np.argsort(self.ids)
        self.wnids = np.squeeze(np.array([x.WNID for x in self.synsets]))[idx]
        self.classnames = np.squeeze(np.array([x.words for x in self.synsets]))[idx]
        self.class_used = np.zeros(len(self.classnames), dtype=bool)
        #from IPython import embed
        #embed()

    def winid_classid_2010(self):
        lines = []
        def from_path(pathid):
            synsets = glob(os.path.join("/home/local/backup/ILSVRC2010/256x256/%s" % pathid, "*"))
            for synset in synsets:
                images = glob(os.path.join(synset, "*"))
                for m in images:
                    if m.endswith(".jpg") or m.endswith(".JPEG"):
                        wnid = os.path.split(os.path.split(m)[0])[1]
                        result = np.where(self.wnids==wnid)
                        #from IPython.core.debugger import Tracer
                        #Tracer()()
                        if len(result[0]) != 1:
                            raise ValueError("Invalid wnid." + str(wnid))
                        klass = self.ids[result[0][0]]
                        self.class_used[klass] = True
                        # -1 for object count is a marker for a pure classification dataset
                        info = [m, "-1", str(klass)]
                        lines.append(info)

            seed(42)
            shuffle(lines)
            return lines

        def write_ds(dir, filename, lines):
            with open(os.path.join(dir,filename), "w") as f:
                f.write(str(len(self.classnames)) + "\n")
                for used, wnid, cls in zip(self.class_used, self.wnids, self.classnames):
                    if not used:
                        cls = "unused"
                    f.write(cls + ", " + wnid + "\n")
                for line in lines:
                    f.write("\t".join(line))
                    f.write("\n")

        train = from_path("train")
        val = from_path("val")

        # remove unused classes at the end
        while not self.class_used[-1]:
            self.class_used = self.class_used[:-1]
            self.classnames = self.classnames[:-1]

        d = "/tmp"
        if not os.path.exists(d):
            os.makedirs(d)

        write_ds(d, "ds_ILSVRC2010_train.txt", train)
        write_ds(d, "ds_ILSVRC2010_val.txt", val)

    def winid_classid(self):
        lines = []
        tarfiles = glob(os.path.join("/home/backup/ILSVRC2012/train", "*"))
        for filename in tarfiles:
            with tarfile.open(filename) as f:
                for m in f.getmembers():
                    if m.name.endswith(".jpg") or m.name.endswith(".JPEG"):
                        wnid = m.name.split("_")[0]
                        downsampled = os.path.join("/home/data/ILSVRC2012/downsampled2", wnid, m.name)
                        if not os.path.exists(downsampled):
                            print "does not exist: ", downsampled, " -- ignoring."
                            continue
                        result = np.where(self.wnids==wnid)
                        if len(result[0]) != 1:
                            raise ValueError("Invalid wnid." + str(wnid))
                        klass = self.ids[result[0][0]]
                        self.class_used[klass] = True
                        # -1 for object count is a marker for a pure classification dataset
                        info = [wnid + "/" + m.name, str(klass)]
                        lines.append(info)

        seed(42)
        shuffle(lines)
        split = 6144
        val   = lines[:split]
        train = lines[split:]

        def write_ds(dir, filename, lines):
            with open(os.path.join(dir,filename), "w") as f:
                f.write(str(len(self.classnames)) + "\n")
                for used, wnid, cls in zip(self.class_used, self.wnids, self.classnames):
                    if not used:
                        cls = "unused"
                    f.write(cls + ", " + wnid + "\n")
                for line in lines:
                    f.write("\t".join(line))
                    f.write("\n")

        # remove unused classes at the end
        while not self.class_used[-1]:
            self.class_used = self.class_used[:-1]
            self.classnames = self.classnames[:-1]

        d = "/tmp"
        if not os.path.exists(d):
            os.makedirs(d)

        write_ds(d, "ds_ILSVRC2014_train.txt", train)
        write_ds(d, "ds_ILSVRC2014_val.txt", val)

def handle_tarfile(filename):
    dstdir = filename.replace("tars", "downsampled")[:-4]  # remove ".tar"
    if not os.path.exists(dstdir):
        os.makedirs(dstdir)
    print "Handling tarfile: ", filename, " --> ", dstdir
    with tarfile.open(filename) as f:
        for m in f.getmembers():
            try:
                img = cv2.imdecode(np.fromstring(f.extractfile(m).read(),dtype=np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
                fact = 256. / min(img.shape[:2])  # ensure shorter dimension is 256 long
                img = cv2.resize(img, (int(np.round(fact * img.shape[1])), int(np.round(fact * img.shape[0]))))
                dstfilename = os.path.join(dstdir, m.name)
                cv2.imwrite(dstfilename, img)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print "Failure for image:", file, ": ", str(e)


def downsample_all_missing(base="/home/local/backup/ILSVRC2011/tars"):
    tarfiles = glob(os.path.join(base, "*"))
    Parallel(n_jobs=-1)(delayed(handle_tarfile)(f) for f in tarfiles)


if __name__ == "__main__":

    #downsample_all_missing()
    ind = ImageNetData()
    #ind.winid_classid_2010()
    ind.winid_classid()
