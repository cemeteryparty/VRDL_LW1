""" Image files TO Numerical files  

Load and transform training and testing images into numerical files to reduce
	time cost of loading image.  
"""
from scipy.io import loadmat, savemat
from PIL import Image
import numpy as np
import os, sys

""" Load Classes Info """
classes = []
fd = open(BASE_DIR + "classes.txt", "r")
for line in fd.readlines():
	classes.append(line.strip())
fd.close()

""" Training image to numerical data """
fd = open("training_labels.txt", "r")
content = fd.readlines()
fd.close()
images = np.zeros((len(content), 224, 224, 3))
labels = np.zeros((len(content), ))
for i in range(len(content)):
	line = content[i].strip()
	im = Image.open("training_images/" + line[:8])
	im = im.resize((224, 224), Image.BILINEAR)
	images[i] = np.array(im)
	labels[i] = classes.index(line[9:])
print(images.shape, labels.shape)

for c in range(200):
	print("class:", classes[c])
	idx = 0
	Cimages = np.zeros((15, 224, 224, 3), dtype=np.float32)
	Clabels = np.zeros((15, ), dtype=np.uint8)
	for i in range(3000):
		if labels[i] == c:
			Cimages[idx] = images[i]
			Clabels[idx] = labels[i]
			idx += 1
	Clabels = utils.to_categorical(Clabels, num_classes=200, dtype=np.int8)
	print(Cimages.shape, Clabels.shape)
	class_dict = {"x_data": Cimages, "y_data": Clabels}
	savemat(f"m224/classes{c}.mat", class_dict)


""" Testing image to numerical data 

Provide in gdrive resource folder already
"""
fd = open("testing_img_order.txt", "r")
content = fd.readlines()
fd.close()
x_test = np.zeros((len(content), 224, 224, 3), dtype=np.float32)
for i in range(len(content)):
	line = content[i].strip()
	im = Image.open("testing_images/" + line)
	im = im.resize((224, 224), Image.BILINEAR)
	x_test[i] = np.array(im)
savemat("testing224.mat", {"x_data": x_test})

