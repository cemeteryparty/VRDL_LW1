#!/opt/anaconda3/envs/pylab/bin/python3
import sys, os
if "include" not in sys.path:
	sys.path.insert(1, "include") # IMPORTANT
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras.models import load_model
from scipy.io import loadmat, savemat
import numpy as np
import math, time

BASE_DIR = "/mnt/left/phlai_DATA/lab5008/"
basepath = "/mnt/left/phlai_DATA/tmp/"
ONSET = "xce-mft1450"
ep = sys.argv[1]
batch_size = 32
nb_batch = 32

""" Load Info """
classes = []
fd = open(BASE_DIR + "classes.txt", "r")
for line in fd.readlines():
	classes.append(line.strip())
fd.close()
""" Form training and validation data """
data_order = np.arange(15)
#np.random.seed(1450)
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
x_valid = np.zeros((3000, 250, 200, 3), dtype=np.float32)
y_valid = np.zeros((3000, 200), dtype=np.uint8)
idx = 0
for c in range(200):
	mat = loadmat(BASE_DIR + f"mat/classes{c}.mat")
	for i in data_order:#[10:]
		x_valid[idx] = mat["x_data"][i]#np.flip(mat["x_data"][i], axis=1)
		y_valid[idx] = mat["y_data"][i]
		idx += 1
print("train:", x_train.shape, y_train.shape)
print("valid:", x_valid.shape, y_valid.shape)

# logs = {"iter": [], "acc": [], "val_acc": []}
try:
	bound = np.load("output/bound.npy")
except FileNotFoundError:
	bound = np.zeros((3,), dtype=np.float64)

tra_order = np.arange(x_train.shape[0])
np.random.shuffle(tra_order)
val_order = np.arange(x_valid.shape[0])
np.random.shuffle(val_order)
x_train = x_train[tra_order]; y_train = y_train[tra_order]
x_valid = x_valid[val_order]; y_valid = y_valid[val_order]
TimeStampStart = time.time()
for bs in range(nb_batch):
	mpath = basepath + f"{ONSET}_ep{ep}_{bs}.h5py"
	if os.path.exists(mpath):
		model = load_model(mpath)
		_, acc = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
		_, vacc = model.evaluate(x_valid, y_valid, batch_size=batch_size, verbose=0)
		print("\rEp{} {}/{} - {:.4f}s - acc = {} - val_acc = {}".format(
			ep, bs + 1, nb_batch, time.time() - TimeStampStart, acc, vacc), end=""
		)
		if acc > bound[0]:
			bound[0] = acc
			model.save(f"models/{ONSET}_tra.h5py")
		if vacc > bound[1]:
			bound[1] = vacc
			model.save(f"models/{ONSET}_val.h5py")
		if (acc + vacc) / 2.0 > bound[2]:
			bound[2] = (acc + vacc) / 2.0
			model.save(f"models/{ONSET}_best.h5py")
## ENDLOOP

print("\n[Finished]")
print("Ep{} - {:.4f}s - acc = {} - val_acc = {}".format(ep, time.time() - TimeStampStart, bound[0], bound[1]))
np.save("output/bound.npy", bound)
exit(0)

## FI
try:
	fd = open("output/bound.log", "r")
	bound = np.float64(fd.read().strip())
	fd.close()
except FileNotFoundError:
	bound = np.inf

try:
	mat = loadmat("output/metrics.mat")
	logs = {"iter": mat["iter"].squeeze(), "loss": mat["loss"].squeeze()}
except FileNotFoundError:
	logs = {"iter": [], "loss": []}

tra_order = np.arange(x_train.shape[0]); 
nb_batch = 38 #math.ceil(x_train.shape[0] / batch_size)
np.random.shuffle(tra_order)
x_train_ = x_train[tra_order]; y_train_ = y_train[tra_order]
TimeStampStart = time.time()
for bs in range(nb_batch):
	mpath = f"/mnt/left/phlai_DATA/tmp/{ONSET}_ep{ep}_{bs}.h5py"
	if not os.path.exists(mpath):
		continue
	model = load_model(mpath)
	loss, _ = model.evaluate(x_train_, y_train_, batch_size=batch_size, verbose=0)
	#loss, _ = model.evaluate(x_valid, y_valid, batch_size=batch_size, verbose=0)

	print("\rEp{} {}/{} - {:.4f}s loss = {}".format(ep, bs + 1, nb_batch, time.time() - TimeStampStart, loss), end="")
	logs["loss"] = np.append(logs["loss"], loss)
	logs["iter"] = np.append(logs["iter"], int(ep) * nb_batch + bs)
	if loss < bound:
		bound = loss
		os.system("echo {:.15f} > output/bound.log".format(loss))
		model.save(f"models/{ONSET}_tra.h5py")
		#model.save(f"models/{ONSET}_val.h5py")
	## FI
## DONE
print("\n")
savemat("output/metrics.mat", logs)
