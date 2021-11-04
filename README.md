# VRDL LabWork001

# Top_Reproduce `answer.txt` (Using inference.py)

Download models and `inference.py` from below link first.

gdrive resource link: https://drive.google.com/drive/folders/1wp9Vnk6ovlME4CUqmdC9xDcDlRPGjXbB

## GDrive Structure

 - resource (9 models, 1 python provided by LINK)
 	- rn50-ft1600.h5py
 	- rn50-ft1641.h5py
 	- rn50-ft2000.h5py
 	- rn50-ft2001.h5py
 	- rn50-ft2003.h5py
 	- rn50-ft2004.h5py
 	- rn50-ft2006.h5py
 	- rn50-ft2008.h5py
 	- rn50-ft2009.h5py
 	- `inference.py`
 	- `classes.txt (Download from Competition)`
 	- `testing_img_order.txt (Download from Competition)`
 	- testing_images/ (Download from Competition)

Notice: `classes.txt`, `testing_img_order.txt`, testing_images folder are NOT provided in GoogleDrive. Please download these files and folder from 2021 VRDL HW1 Competition and put it under the same directory with `inference.py` and models.

Generate `answer.txt`, checking the directory structure before executing
```sh
$ python3 inference.py
```

# Repo Structure

- VRDL_LW1/
	- m224/
	- models/
		- (model_name).h5py
	- src/
		- `img2num.py`
		- `lab1_ansgen.py`
		- `lab1_ft_train.py`
		- `lab1_train.py`
		- `inference.py`
	- testing_images (Download from Competition)/
	- training_images (Download from Competition)/
	- `classes.txt`
	- `testing_img_order.txt`
	- `training_labels.txt`

# Environment

- Python     3.7.10
	- scikit-learn           0.24.2
	- scipy                  1.6.2
	- numpy                  1.20.2
	- tensorflow             2.2.0
	- PIL                    7.1.2

