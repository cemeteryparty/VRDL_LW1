from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input
from scipy.io import loadmat
from scipy.io import savemat
from PIL import Image
import numpy as np
import sys
import os

""" Load Info """
classes = []
fd = open("classes.txt", "r")
for line in fd.readlines():
    classes.append(line.strip())
fd.close()
""" Load testing image order """
fd = open("testing_img_order.txt", "r")
content = fd.readlines()
fd.close()

""" Load data """
if not os.path.exists("testing224.mat"):
    x_test = np.zeros((len(content), 224, 224, 3), dtype=np.float32)
    for i in range(len(content)):
        line = content[i].strip()
        im = Image.open("testing_images/" + line)
        im = im.resize((224, 224), Image.BILINEAR)
        x_test[i] = np.array(im)
    savemat("testing224.mat", {"x_data": x_test})

x_test = loadmat("testing224.mat")["x_data"]
x_mirror = loadmat("testing224.mat")["x_data"]
for i in range(x_mirror.shape[0]):
    x_mirror[i] = np.flip(x_mirror[i], axis=1)
x_test = preprocess_input(x_test)
x_mirror = preprocess_input(x_mirror)

model_paths = []
for fpath in os.listdir():
    if ".h5py" in fpath:
        print(fpath)
        model_paths.append(fpath)
models = [load_model(mpath) for mpath in model_paths]

""" model poll """
prob_mat = np.zeros((len(content), 200), dtype=np.float64)
for model in models:
    y_pred = model.predict(x_test, batch_size=32)
    yi = y_pred.argmax(axis=1)
    yp = y_pred.max(axis=1)

    y_pred2 = model.predict(x_mirror, batch_size=32)
    yi2 = y_pred2.argmax(axis=1)
    yp2 = y_pred2.max(axis=1)

    for i in range(len(content)):
        prob_mat[i][yi[i]] += yp[i]
        prob_mat[i][yi2[i]] += yp2[i]

""" Generate answer.txt """
fd = open("answer.txt", "a")
prob_mat = prob_mat.argmax(axis=1)
for i in range(len(content)):
    line = content[i].strip()
    label = classes[prob_mat[i]]
    fd.write("{} {}\n".format(line, label))
fd.close()

print("[Finished]")
