import os
import numpy as np
from random import shuffle
from scipy.io import loadmat
import matplotlib as plt
from scipy.ndimage import zoom
import Image
from joblib import Parallel, delayed

class ImageNetData(object):

    def __init__(self):
        self.meta_path = "/home/local/backup/ILSVRC2011/ILSVRC2011_devkit-2.0/data"
        self.meta_data = loadmat(os.path.join(self.meta_path, "meta.mat"), struct_as_record=False)

        self.synsets = np.squeeze(self.meta_data['synsets'])
        self.ids = np.squeeze(np.array([x.ILSVRC2011_ID for x in self.synsets]))

        self.wnids = np.squeeze(np.array([x.WNID for x in self.synsets]))
        from IPython import embed
        embed()

    def winid_classid(self):
        lines = []
        for root, dirs, files in os.walk("/home/local/backup/ILSVRC2011/unpacked"):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".JPEG"):

                    wnid = file.split("_")[0]

                    downsampled = os.path.join("/home/local/backup/ILSVRC2011/downsampled", wnid, file)
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
                for line in lines:
                    f.write("\t".join(line))
                    f.write("\n")
        d = "/tmp"
        if not os.path.exists(d):
            os.makedirs(d)

        write_ds(d, "ds_ILSVRC2011_train.txt", train)
        write_ds(d, "ds_ILSVRC2011_test.txt", val)


def downsample_image(filename):
    img = Image.open(filename).convert("RGB")
    img = np.array(img)
    fact = 256. / max(img.shape[:2])
    img = zoom(img, (fact, fact, 1))
    return img

def handle_file(srcdir, dstdir, file):
    if file.endswith(".jpg") or file.endswith(".JPEG"):
        src = os.path.join(srcdir, file)
        dst = os.path.join(dstdir, file)
        if os.path.exists(dst):
            return
        print "downsampling ", src
        try:
            img = downsample_image(src)
            img = Image.fromarray(img)
            img.save(dst)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print "Failure for image:", file, ": ", str(e)

def downsample_all_missing(base="/home/local/backup/ILSVRC2011/unpacked", size=256):
    for srcdir, dirs, files in os.walk(base):
        dstdir = srcdir.replace("unpacked", "downsampled")
        if not os.path.exists(dstdir):
            os.makedirs(dstdir)
        Parallel(n_jobs=1)(delayed(handle_file)(srcdir, dstdir, f) for f in files)


if __name__ == "__main__":

    ind = ImageNetData()
    ind.winid_classid()
    #downsample_all_missing()
