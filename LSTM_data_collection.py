# USAGE
# python detect_face_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
import pickle
import random

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



from PIL import Image




person = ["F01", "F02", "F04", "F05", "F06", "F07", "F08", "F11", "M01", "M02", "M04", "M07", "M08"]
prob = ["phrases"]
label = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
instance = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
min_sequence_length = 4
temp_path = "/Users/michaeldu/Desktop/cs221/project/detect-face-parts/dataset"
new_path = "/Users/michaeldu/Desktop/cs221/project/detect-face-parts/cropped_imgs"

X = []
Y = []
max_seq_length = 0

# AVERAGE IMAGES
for p1 in person:
	print(p1)
	for p2 in prob:
		for l in label:
			for k in instance:
				X_train = np.zeros((50, 50, 3), dtype=np.float64)
				count = 0
				curr_path = temp_path + "/" + p1 + "/" + p2 + "/" + l + "/" + k
				new_path = "/Users/michaeldu/Desktop/cs221/project/detect-face-parts/cropped_imgs_2"+ "/" + p1 + "/" + p2 + "/" + l + "/" + k

				instance_data = []

				for filename in os.listdir(curr_path):
					if filename.endswith(".jpg"):
						# load the input image, resize it, and convert it to grayscale
						image = cv2.imread(os.path.join(curr_path, filename))
						image = imutils.resize(image, width=500)
						gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

						# detect faces in the grayscale image
						rects = detector(gray, 1)

						# loop over the face detections
						for (i, rect) in enumerate(rects):
							# determine the facial landmarks for the face region, then
							# convert the landmark (x, y)-coordinates to a NumPy array
							shape = predictor(gray, rect)
							shape = face_utils.shape_to_np(shape)

							# specifies mouth region for detection
							name = 'mouth'
							i = 48
							j = 68

							# clone the original image so we can draw on it, then
							# display the name of the face part on the image
							clone = image.copy()
							cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
								0.7, (0, 0, 255), 2)

							# loop over the subset of facial landmarks, drawing the
							# specific face part
							for (x, y) in shape[i:j]:
								cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

							# extract the ROI of the face region as a separate image
							(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
							roi = image[y:y + h, x:x + w]

							#specifies width of image, so height will vary
							roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

							# appends the height of mouth region to instance data
							instance_data.append(roi.shape[0])

				# appends heights for instance to X and label of instance to Y
				X.append(instance_data)
				Y.append(int(l)-1)

				# keeps track of max seq length for validation
				max_seq_length = max(len(instance_data), max_seq_length)


print("max seq length is", max_seq_length)
#writes array to pickle file for transferability 
pickle.dump(X, open("data_X.p", "wb"))
pickle.dump(Y, open("data_Y.p", "wb"))
