#!/opt/anaconda3/envs/pylab/bin/python3
import sys, os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
if "include" not in sys.path:
	sys.path.insert(1, "include") # IMPORTANT
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score, confusion_matrix
from scipy.io import loadmat, savemat
import pandas as pd
import numpy as np
import math, time

BASE_DIR = "/mnt/left/phlai_DATA/lab5008/"
model_path = "models/xce_ft1429m.h5py"
classes = []
fd = open(BASE_DIR + "classes.txt", "r")
for line in fd.readlines():
	classes.append(line.strip())
fd.close()

""" Form training data """
# x_train = np.zeros((3000, 250, 200, 3), dtype=np.float32)
# x_mirror = np.zeros((3000, 250, 200, 3), dtype=np.float32)
# y_train = np.zeros((3000, 200), dtype=np.uint8)
# idx = 0
# for c in range(200):
# 	mat = loadmat(BASE_DIR + f"mat/classes{c}.mat")
# 	for i in data_order:
# 		x_train[idx] = mat["x_data"][i]
# 		x_mirror[idx] = np.flip(mat["x_data"][i], axis=1)
# 		y_train[idx] = mat["y_data"][i]
# 		idx += 1
# print("train:", x_train.shape, y_train.shape)
# print("mirror:", x_mirror.shape, y_train.shape)

# model = load_model(model_path)
# y_pred = model.predict(x_train, batch_size=32).argmax(axis=1)
# print("Original:", model.evaluate(x_train, y_train, batch_size=32, verbose=0))
# y_pred = model.predict(x_mirror, batch_size=32).argmax(axis=1)
# print("Mirror:", model.evaluate(x_mirror, y_train, batch_size=32, verbose=0))

x_train = np.zeros((6000, 250, 200, 3), dtype=np.float32)
y_train = np.zeros((6000, 200), dtype=np.uint8)
idx = 0
for c in range(200):
	mat = loadmat(BASE_DIR + f"mat/classes{c}.mat")
	for i in range(15):
		x_train[idx] = mat["x_data"][i]
		x_train[idx + 1] = np.flip(mat["x_data"][i], axis=1)
		y_train[idx] = mat["y_data"][i]
		y_train[idx + 1] = mat["y_data"][i]
		idx += 2
model = load_model(model_path)
y_pred = model.predict(x_train, batch_size=32).argmax(axis=1)
fscore = f1_score(y_train.argmax(axis=1), y_pred, average=None)
np.save(model_path[7:18] + ".npy", fscore)
