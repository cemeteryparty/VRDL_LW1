import sys, os
from tensorflow import keras
from tensorflow.keras import layers, utils
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from scipy.io import loadmat
import numpy as np
import math, time

""" Function Define """
class MyModelCkpt(keras.callbacks.Callback):
	def __init__(self, filepath, monitor="val_loss", mode="min", verbose=0):
		super().__init__()
		self.filepath = filepath
		self.monitor = monitor
		if mode == "min":
			self.mode = mode; self.bound = np.inf
		elif mode == "max":
			self.mode = mode; self.bound = -np.inf
		else:
			raise ValueError("mode in MyModelCkpt can only be \'min\' or \'max\'")
		self.verbose = verbose
	def on_epoch_end(self, epoch, logs=None):
		if ((self.mode == "min" and logs[self.monitor] < self.bound) or 
			(self.mode == "max" and logs[self.monitor] > self.bound)):
			if self.verbose:
				print("Epoch {} : {} improved from {} to {}\n".format(
					epoch, self.monitor, self.bound, logs[self.monitor])
				)
			self.bound = logs[self.monitor]
			self.model.save(self.filepath)
		else:
			pass
def ResNet50_T(include_top=False, input_shape=None, pooling="avg", classes=1000):
	rn50 = ResNet50(include_top=include_top, weights="imagenet", input_shape=input_shape, pooling=pooling, classes=classes)
	outputs = layers.Dense(units=classes, activation="softmax", name="predictions")(rn50.output)
	return Model(inputs=rn50.input, outputs=outputs, name="resnet50")

""" Load Info """
classes = []
fd = open("classes.txt", "r")
for line in fd.readlines():
	classes.append(line.strip())
fd.close()

""" main """
data_order = np.arange(15)
np.random.shuffle(data_order)
x_train = np.zeros((3000, 224, 224, 3), dtype=np.float32)
y_train = np.zeros((3000, 200), dtype=np.uint8)
idx = 0
for c in range(200):
	mat = loadmat(f"m224/classes{c}.mat")
	for i in data_order[:12]:
		x_train[idx] = mat["x_data"][i]
		y_train[idx] = mat["y_data"][i]
		idx += 1
x_train = x_train[:idx]; y_train = y_train[:idx]
print(idx, x_train.shape, y_train.shape)
x_valid = np.zeros((3000, 224, 224, 3), dtype=np.float32)
y_valid = np.zeros((3000, 200), dtype=np.uint8)
idx = 0
for c in range(200):
	mat = loadmat(f"m224/classes{c}.mat")
	for i in data_order[12:]:
		x_valid[idx] = mat["x_data"][i]
		y_valid[idx] = mat["y_data"][i]
		idx += 1
x_valid = x_valid[:idx]; y_valid = y_valid[:idx]
print(idx, x_valid.shape, y_valid.shape)
tra_datagen = ImageDataGenerator(
	preprocessing_function=preprocess_input,
	rotation_range=10, width_shift_range=0.2,
	height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True
)
val_datagen = ImageDataGenerator(
	preprocessing_function=preprocess_input,
	rotation_range=10, width_shift_range=0.2,
	height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True
)
trainset = tra_datagen.flow(x_train, y_train, batch_size=32)
validset = val_datagen.flow(x_valid, y_valid, batch_size=32)

""" Main Training Process """
model = ResNet50_T(input_shape=(224, 224, 3), classes=200)
opt = Adam(learning_rate=ExponentialDecay(1e-3, decay_steps=63, decay_rate=0.96))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])
ONSET = f"rn50-1500"
vloss_ckpt = MyModelCkpt(f"models/{ONSET}.h5py", monitor="val_loss", mode="min", verbose=0)
history = model.fit(trainset, epochs=40, verbose=2, callbacks=[loss_ckpt], 
	validation_data=validset)

print("[Finished]")
