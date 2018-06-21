import copy
import os
from subprocess import call

import numpy as np
import sklearn
import sklearn.cross_validation
import sklearn.linear_model

import h5py

TARGET_PATH='~/data/'

print("Downloading...")
if not os.path.exists("cifar-10-python.tar.gz"):
    call(
        "wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        shell=True
    )
    print("Downloading done.\n")
else:
    print("Dataset already downloaded. Did not download twice.\n")


print("Extracting...")
if not os.path.exists(TARGET_PATH):
    call(
        "tar -zxvf cifar-10-python.tar.gz -C %s" % os.path.expanduser(TARGET_PATH),
        shell=True
    )
    print("Extracting successfully done to {}.".format(TARGET_PATH))
else:
    print("Dataset already extracted. Did not extract twice.\n")

print("Removing...")
# call("rm cifar-10-python.tar.gz", shell=True)