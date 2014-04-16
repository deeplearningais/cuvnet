import os
from glob import glob
import numpy as np
from random import shuffle
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
        self.meta_path = "/home/local/backup/ILSVRC2011/ILSVRC2011_devkit-2.0/data"
        self.meta_data = loadmat(os.path.join(self.meta_path, "meta.mat"), struct_as_record=False)

        self.synsets = np.squeeze(self.meta_data['synsets'])
        self.ids = np.squeeze(np.array([x.ILSVRC2011_ID for x in self.synsets]))

        self.wnids = np.squeeze(np.array([x.WNID for x in self.synsets]))
        self.classnames = np.squeeze(np.array([x.words for x in self.synsets]))
        #from IPython import embed
        #embed()

    def winid_classid(self):
        lines = []
        tarfiles = glob(os.path.join("/home/local/backup/ILSVRC2011/tars", "*"))
        for filename in tarfiles:
            with tarfile.open(filename) as f:
                for m in f.getmembers():
                    if m.name.endswith(".jpg") or m.name.endswith(".JPEG"):
                        wnid = m.name.split("_")[0]
                        downsampled = os.path.join("/home/local/backup/ILSVRC2011/downsampled", wnid, m.name)
                        if not os.path.exists(downsampled):
                            print "does not exist: ", downsampled, " -- ignoring."
                            continue
                        result = np.where(self.wnids==wnid)
                        if len(result[0]) == 0:
                            raise ValueError("Invalid wnid.")
                        # -1 for object count is a marker for a pure classification dataset
                        info = [downsampled, "-1", str(result[0][0])]
                        lines.append(info)

        shuffle(lines)
        split = len(lines) * 1 / 5
        val   = lines[:split]
        train = lines[split:]

        def write_ds(dir, filename, lines):
            with open(os.path.join(dir,filename), "w") as f:
                f.write(str(len(self.classnames)) + "\n")
                for cls in self.classnames:
                    f.write(cls + "\n")
                for line in lines:
                    f.write("\t".join(line))
                    f.write("\n")
        d = "/tmp"
        if not os.path.exists(d):
            os.makedirs(d)

        write_ds(d, "ds_ILSVRC2011_train.txt", train)
        write_ds(d, "ds_ILSVRC2011_test.txt", val)

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
    ind.winid_classid()
