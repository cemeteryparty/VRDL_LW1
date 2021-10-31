#!/opt/anaconda3/envs/pylab/bin/python3
import sys, os

from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras.models import load_model
from scipy.io import loadmat, savemat
import numpy as np

BASE_DIR = "/mnt/left/phlai_DATA/lab5008/"

classes = []
fd = open(BASE_DIR + "classes.txt", "r")
for line in fd.readlines():
	classes.append(line.strip())
fd.close()

x_test = loadmat(BASE_DIR + "data/testing.mat")["x_data"]
x_mirror = loadmat(BASE_DIR + "data/testing.mat")["x_data"]
for i in range(x_mirror.shape[0]):
	x_mirror[i] = np.flip(x_mirror[i], axis=1)
model = load_model("models/rn50-mo1500_val.h5py")
y_pred = model.predict(x_test, batch_size=32)
y1 = y_pred.max(axis=1)
y_pred = y_pred.argmax(axis=1)
print(y_pred.shape, y1.shape)

y_pred2 = model.predict(x_mirror, batch_size=32)
y2 = y_pred2.max(axis=1)
y_pred2 = y_pred2.argmax(axis=1)
print(y_pred2.shape, y2.shape)

fd = open(BASE_DIR + "testing_img_order.txt", "r")
content = fd.readlines()
fd.close()
for i in range(len(content)):
	line = content[i].strip() # image name
	if y1[i] > y2[i]:
		label = classes[y_pred[i]]
	else:
		label = classes[y_pred2[i]]
	os.system("echo {} >> answer.txt".format(f"{line} {label}"))

print("[Finished]")