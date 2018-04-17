# Libraries for Part 1
import dicom
import os
import pandas as pd
# Libraries for Part 2
import matplotlib.pyplot as plt

import cv2
import numpy as np

import math

# Part 1a: Import that CT scan data and the csv file with the corresponding diagnoses
data_dir = '/Users/danielhardej/Documents/Sentdex_tutorials/Lung_Cancer_Detection/Preprocessing_Tut/input/sample_images/'
patients = os.listdir(data_dir)
labels = pd.read_csv('/Users/danielhardej/Documents/Sentdex_tutorials/Lung_Cancer_Detection/Preprocessing_Tut/input/stage1_labels.csv', index_col=0)

# print(labels_df.head(5))	# returns first n rows of labels_df

# # Part 1b: 
# for patient in patients[1:2]:
# 	label = labels_df.get_value(patient, 'cancer')
# 	path = data_dir + patient
# 	# read DICOM files from each patient from data_dir folder
# 	# 's' denotes the file containing the DICOM images for a given patient
# 	slices = [dicom.read_file(path+'/'+s) for s in os.listdir(path)]
# 	slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
# 	if label == 1:
# 		diagnosis = 'Has cancer'
# 	else:
# 		diagnosis = 'Does not have cancer'
# 	# Print the number of slices in the 3d med image for the patient and the label indicating whether the patient has cancer and the size of the array representing the slice	
# 	print('Patient ID: ', patient, 'Number of slices: ', len(slices), 'Size of slice: ', slices[0].pixel_array.shape, 'Diagnosis: ', diagnosis)
# 	#print(diagnosis, slices[0])	# display the metadata for the DICOM image file

# # Part 2: This part of the program provides a visualization of the data in the DICOM files
# for patient in patients[1:2]:
# 	# As in the previous function, call the patient DICOM data and the corresponding diagnosis
# 	label = labels_df.get_value(patient, 'cancer')
# 	path = data_dir + patient
# 	slices = [dicom.read_file(path+'/'+s) for s in os.listdir(path)]
# 	slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
# 	# Show the first CT scan slice of the 3D image
# 	# plt.imshow(slices[0].pixel_array)
# 	# plt.show()

# IMG_PX_SIZE = 150

# # Part 3: Plot the first 'n' CT scan slices for a given patient or set of patients
# for patient in patients[1:2]:
# 	label = labels_df.get_value(patient, 'cancer')
# 	path = data_dir + patient
# 	slices = [dicom.read_file(path+'/'+s) for s in os.listdir(path)]
# 	slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
# 	fig = plt.figure()
# 	for num, each_slice in enumerate(slices[:12]):
# 		y = fig.add_subplot(3,4,num+1)
# 		new_img = cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE))
# 		y.imshow(new_img)
# 	plt.show()

# Part 4: This funtion modifies the set of CT scans for each patient
# so that all sets of scans consist of a uniform number of slices
def chunks(l,n):
	"""Yield successive n-sized chunks from a list l"""
	for i in range(0, len(l), int(n)):
		yield l[i:i + int(n)]

def mean(l):
	return sum(l) / len(l)

IMG_SIZE_PX = 50
HM_SLICES = 20

# # NB - this is a really bad way of homogenizing the number of slices in each patients set of CT scans!!!
# for patient in patients[1:]:
# 	try:
# 		label = labels_df.get_value(patient, 'cancer')
# 		path = data_dir + patient
# 		slices = [dicom.read_file(path+'/'+s) for s in os.listdir(path)]
# 		slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
# 		new_slices = []

# 		slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]
# 		chunk_sizes = math.ceil(len(slices)/HM_SLICES)

# 		for slice_chunk in chunks(slices, chunk_sizes):
# 			slice_chunk = list(map(mean, zip(*slice_chunk)))
# 			new_slices.append(slice_chunk)
# 		if len(new_slices) == HM_SLICES-1:
# 			new_slices.append(new_slices[-1])
# 		if len(new_slices) == HM_SLICES-2:
# 			new_slices.append(new_slices[-1])
# 			new_slices.append(new_slices[-1])
# 		if len(new_slices) == HM_SLICES+2:
# 			new_val = list(map(mean, zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))
# 			del new_slices[HM_SLICES]
# 			new_slices[HM_SLICES-1] = new_val
# 		if len(new_slices) == HM_SLICES+1:
# 			new_val = list(map(mean, zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))
# 			del new_slices[HM_SLICES]
# 			new_slices[HM_SLICES-1] = new_val

# 		print('Patient: ', patient, 'Number of original slices: ', len(slices), 'Number of new slices: ', len(new_slices))
# 	except Exception as e:
# 		print('Failed to reduce: ', str(e))


def process_data(patient,labels_df,img_px_size=50, hm_slices=20, visualize=False):
    
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    new_slices = []
    slices = [cv2.resize(np.array(each_slice.pixel_array),(img_px_size,img_px_size)) for each_slice in slices]

    chunk_sizes = math.ceil(len(slices) / hm_slices)
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    if len(new_slices) == hm_slices-1:
        new_slices.append(new_slices[-1])

    if len(new_slices) == hm_slices-2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    if len(new_slices) == hm_slices+2:
        new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
        del new_slices[hm_slices]
        new_slices[hm_slices-1] = new_val
        
    if len(new_slices) == hm_slices+1:
        new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
        del new_slices[hm_slices]
        new_slices[hm_slices-1] = new_val

    if visualize:
        fig = plt.figure()
        for num,each_slice in enumerate(new_slices):
            y = fig.add_subplot(4,5,num+1)
            y.imshow(each_slice, cmap='gray')
        plt.show()

    if label == 1: label=np.array([0,1])
    elif label == 0: label=np.array([1,0])
        
    return np.array(new_slices),label

much_data = []
for num,patient in enumerate(patients):
    if num % 100 == 0:
        print(num)
    try:
        img_data,label = process_data(patient,labels,img_px_size=IMG_SIZE_PX, hm_slices=HM_SLICES)
        #print(img_data.shape,label)
        much_data.append([img_data,label])
    except KeyError as e:
        print('This is unlabeled data!')

np.save('muchdata-{}-{}-{}.npy'.format(IMG_SIZE_PX,IMG_SIZE_PX,HM_SLICES), much_data)
