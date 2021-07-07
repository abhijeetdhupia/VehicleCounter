import os 
import cv2 
import yaml 
import time
import numpy as np
from scipy import spatial

# Load the config file
with open(r'configs.yaml') as file:
    configs = yaml.full_load(file)

model_config = configs["model"]["model_config"]
classes_path = configs["dataset"]["classes"]
classes = open(classes_path).read().strip().split("\n")
vehicles_list = configs["dataset"]["vehicles"]
input_vid_path = configs["filepaths"]["input"]
output_vid_path = configs["filepaths"]["output"]
weights = configs["model"]["weights"]
defaultConfidence = configs["more_settings"]["defaultConfidence"]
defaultThreshold = configs["more_settings"]["defaultThreshold"]
GPU = configs["more_settings"]["GPU"]
inputHeight = configs["more_settings"]["width"]
inputWidth = configs["more_settings"]["height"]
history = configs["more_settings"]["history"]

# To save the state of a random function
np.random.seed(33)

# Assigning a random colour to each class label from the COCO dataset
box_colours = np.random.uniform(0, 255, size=(len(classes), 3))

# Overlay the number of vehicles along with the number of each classes on the prediction video
def overlayVehicleCount(frame, vehicle_count, bicycle_count, bus_count, car_count, motorbike_count, train_count, truck_count):
	cv2.putText(
		frame,
		f"Total Vehicles: {str(vehicle_count)}, Bicycle: {str(bicycle_count)}, Bus: {str(bus_count)}, Car: {str(car_count)}, Bike: {str(motorbike_count)}, Truck: {str(truck_count)}", #, Train: {str(train_count)}
		(100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0xFF), 4, cv2.FONT_HERSHEY_TRIPLEX,
		)

# Draw all the detected Boxes along with the class names and corresponding confidence scores 
def drawBoxes(idxs, boxes, classIDs, confidences, frame):
	if len(idxs) > 0:
		for i in idxs.flatten():
			# Getting the x, y, height and width values 
			(x, y) = (boxes[i][0], boxes[i][1]) 
			(w, h) = (boxes[i][2], boxes[i][3])

			# Draw the bounding box along with the labels
			colour = [int(col) for col in box_colours[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
			text = "{}: {:.2f}".format(classes[classIDs[i]].capitalize(), confidences[i])
			text = f"{classes[classIDs[i]].capitalize()}: {confidences[i]:.2f}"
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.6, colour, 2)
			
			# Plot the mid point of the box 
			cv2.circle(frame, (x + (w//2), y+ (h//2)), 2, (0, 0xFF, 0), thickness=2)

# Video writer function to save the predcition video
def VideoWriter(video_width, video_height, videoCap):
	input_video_fps = videoCap.get(cv2.CAP_PROP_FPS)
	codec = cv2.VideoWriter_fourcc(*"MJPG")
	return cv2.VideoWriter(output_vid_path, codec, input_video_fps, (video_width, video_height), True)

# Check the previous frames to see if the identified object was freshly found or detected earlier, if present return True or else return False. 
def checkPreviousFrames(previous_frame_detections, current_box, current_detections):
	centerX, centerY, width, height = current_box
	dist = np.inf
	for i in range(history):
		coordinate_list = list(previous_frame_detections[i].keys())
		if len(coordinate_list) == 0: 
			continue
		# Finding the distance to the closest point and the index
		temporary_distance, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
		if (temporary_distance < dist):
			dist = temporary_distance
			frame_num = i
			coord = coordinate_list[index[0]]

	if (dist > (max(width, height)/2)):
		return False

	# Assigning the old ID to the new dectections
	current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
	return True

# Count the total number of vechicles and the corresponding class
def count_vehicles(idxs, boxes, classIDs, vehicle_count, bicycle_count, bus_count, car_count, motorbike_count, train_count, truck_count, previous_frame_detections, frame):
	current_detections = {}
	if len(idxs) > 0:
		for i in idxs.flatten():
			# Bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			
			centerX = x + (w//2)
			centerY = y+ (h//2)
			
			# Check if the detected label belongs in the vechile list 
			if (classes[classIDs[i]] in vehicles_list):
				current_detections[(centerX, centerY)] = vehicle_count 
				# Check if the vechile has been detected previously or not 
				if (not checkPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections)):	
					if classes[classIDs[i]] == "bicycle": 
						bicycle_count += 1 	
						vehicle_count += 1
					elif classes[classIDs[i]] == "bus": 
						bus_count += 1 
						vehicle_count += 1
					elif classes[classIDs[i]] == "car": 
						car_count += 1 
						vehicle_count += 1	
					elif classes[classIDs[i]] == "motorbike": 
						motorbike_count += 1 
						vehicle_count += 1 
					elif classes[classIDs[i]] == "train": 
						train_count += 1 
						vehicle_count += 1
					elif classes[classIDs[i]] == "truck": 
						truck_count += 1 
						vehicle_count += 1

				# Add the current detection mid-point of box to the list of detected items and Get the ID corresponding to the current detection
				ID = current_detections.get((centerX, centerY))

				# If two closeby objects have the same ID then assign a new ID to the current detection.
				if (list(current_detections.values()).count(ID) > 1):
					current_detections[(centerX, centerY)] = vehicle_count
					vehicle_count += 1 

				# Overlay the Class ID at the center of the box
				cv2.putText(frame, str(ID), (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)

	return vehicle_count, bicycle_count, bus_count, car_count, motorbike_count, train_count, truck_count, current_detections

# Load the pretrained YOLOv3 model trained on the COCO dataset from Darknet 
print("Loading the YOLOv3 model")
model = cv2.dnn.readNetFromDarknet(model_config, weights)

# Use GPU if available 
if GPU:
	model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layers = model.getLayerNames()
layers = [layers[i[0] - 1] for i in model.getUnconnectedOutLayers()]

# Initialize the video
videoCap = cv2.VideoCapture(input_vid_path)
video_width = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Specifying coordinates for a default line 
x1_line, y1_line, x2_line, y2_line = 0, video_height//2, video_width, video_height//2

# Initialization
previous_frame_detections = [{(0,0):0} for i in range(history)]
num_frames, vehicle_count, bicycle_count, bus_count, car_count, motorbike_count, train_count, truck_count = 0, 0, 0, 0, 0, 0, 0, 0
writer = VideoWriter(video_width, video_height, videoCap)

print("Please wait ...")
# Loop over all the video frames 
while True:
	num_frames+= 1
	boxes, confidences, classIDs = [], [], [] 
	vehicle_crossed_line_flag = False 
	(grabbed, frame) = videoCap.read()
	if not grabbed:
		break

	# Forward pass using blob 
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight), swapRB=True, crop=False)
	model.setInput(blob)
	start = time.time()
	layer_outputs = model.forward(layers)
	end = time.time()

	# loop over each of the layer outputs
	for output in layer_outputs:
		# loop over each of the detections
		for i, detection in enumerate(output):
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:] 
			classID = np.argmax(scores)
			confidence = scores[classID]

			# Remove the boxes with low confidence score 
			if confidence > defaultConfidence:
				box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# Updating the boxes, confidence scores and classID lists 
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# Using Non-Maximum Suppression technique to select the best bounding box using the Confidence and Threshold scores
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, defaultConfidence, defaultThreshold)

	# Draw detection box 
	drawBoxes(idxs, boxes, classIDs, confidences, frame)
	vehicle_count, bicycle_count, bus_count, car_count, motorbike_count, train_count, truck_count, current_detections = count_vehicles(idxs, boxes, classIDs, vehicle_count, bicycle_count, bus_count, car_count, motorbike_count, train_count, truck_count, previous_frame_detections, frame)

	# Display Vehicle Count if a vehicle has passed the line 
	overlayVehicleCount(frame, vehicle_count, bicycle_count, bus_count, car_count, motorbike_count, train_count, truck_count)

    # Saving the video
	writer.write(frame)
	cv2.imshow('Prediction Video', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		print("Process interrupted ... ")
		break	
	
	previous_frame_detections.pop(0)
	previous_frame_detections.append(current_detections)

print("Saving the Prediction Video")
writer.release()
videoCap.release()