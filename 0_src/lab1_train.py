#!/opt/anaconda3/envs/pylab/bin/python3
import sys, os
if "include" not in sys.path:
	sys.path.insert(1, "include") # IMPORTANT
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
# from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
import pandas as pd
import numpy as np

BASE_DIR = "/mnt/left/phlai_DATA/lab5008/"

classes = []
fd = open(BASE_DIR + "classes.txt", "r")
for line in fd.readlines():
	classes.append(line.strip())
fd.close()

from Keruns import SimpleKeTrainras, MyModelCkpt
from KnownStructure import Xception_M, NASNetLarge_T

data_order = np.arange(15)
np.random.seed(1450)
np.random.shuffle(data_order)
x_train = np.zeros((3000, 250, 200, 3), dtype=np.float32)
y_train = np.zeros((3000, 200), dtype=np.uint8)
idx = 0
for c in range(200):
	mat = loadmat(BASE_DIR + f"mat/classes{c}.mat")
	for i in data_order:#[:10]
		x_train[idx] = np.flip(mat["x_data"][i], axis=1)
		y_train[idx] = mat["y_data"][i]
		idx += 1
# print("\tcount:", idx)
print("train:", x_train.shape, y_train.shape)
"""
x_valid = np.zeros((1000, 250, 200, 3), dtype=np.float32)
y_valid = np.zeros((1000, 200), dtype=np.uint8)
idx = 0
for c in range(200):
	mat = loadmat(BASE_DIR + f"mat/classes{c}.mat")
	for i in data_order[10:]:
		x_valid[idx] = np.flip(mat["x_data"][i], axis=1)
		y_valid[idx] = mat["y_data"][i]
		idx += 1
# print("\tcount:", idx)
print("valid:", x_valid.shape, y_valid.shape)
"""

# model = load_model("models/xce-remft1450_tra.h5py")
# y_pred = model.predict(x_train, batch_size=32).argmax(axis=1)
# print("M1:\n\tTrainEva", model.evaluate(x_train, y_train, batch_size=32, verbose=0))
# y_pred = model.predict(x_valid, batch_size=32).argmax(axis=1)
# print("\tValidEva", model.evaluate(x_valid, y_valid, batch_size=32, verbose=0))
# model = load_model("models/xce-remft1450_val.h5py")
# y_pred = model.predict(x_train, batch_size=32).argmax(axis=1)
# print("M2:\n\tTrainEva", model.evaluate(x_train, y_train, batch_size=32, verbose=0))
# y_pred = model.predict(x_valid, batch_size=32).argmax(axis=1)
# print("\tValidEva", model.evaluate(x_valid, y_valid, batch_size=32, verbose=0))
# exit(0)

model = load_model("models/xce-remft1450.h5py")
opt = Adam(learning_rate=ExponentialDecay(5e-4, decay_steps=56, decay_rate=0.96))
# model = Xception_M(input_shape=x_train.shape[1:], classes=200)
# opt = Adam(learning_rate=ExponentialDecay(1e-3, decay_steps=32, decay_rate=0.95))
model.summary()
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])
ONSET = "xce-mft1450"
# loss_ckpt = MyModelCkpt(f"models/{ONSET}_tra.h5py", monitor="acc", mode="max", verbose=0)
# vloss_ckpt = MyModelCkpt(f"models/{ONSET}_val.h5py", monitor="val_acc", mode="max", verbose=0)
# history = model.fit(x_train, y_train,
# 	epochs=30, batch_size=32, shuffle=True, validation_data=(x_valid, y_valid), 
# 	verbose=2, callbacks=[loss_ckpt, vloss_ckpt])
# savemat(f"output/{ONSET}.mat", history.history)

kt = SimpleKeTrainras(model, basepath = "/mnt/left/phlai_DATA/tmp/")
kt.train(x_train, y_train, batch_size=32, epochs=30, shuffle=True, PREFIX=ONSET)
