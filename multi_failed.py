import cv2
import os

import csv
import sys
import time

def create_tracker(tracker_type):
    if tracker_type == 0:
        return cv2.legacy.TrackerBoosting.create()
    if tracker_type == 1:
        return cv2.legacy.TrackerMIL.create()
    if tracker_type == 2:
        return cv2.legacy.TrackerKCF.create()
    if tracker_type == 3:
        return cv2.legacy.TrackerTLD.create()
    if tracker_type == 4:
        return cv2.legacy.TrackerMedianFlow.create()
    if tracker_type == 5:
        return cv2.TrackerCSRT.create()
    if tracker_type == 6:
        return cv2.legacy.TrackerMOSSE.create()

    """
    if tracker_type == 7:
        return cv2.TrackerDaSiamRPN.create()
    if tracker_type == 8:
        return cv2.TrackerGOTURN.create()
    if tracker_type == 9:
        return cv2.TrackerCSRT.create()
    """
def is_centroid_overlapping(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Compute the centroid of rect2
    centroid_x2 = x2 + w2 // 2
    centroid_y2 = y2 + h2 // 2

    # Check if the centroid of rect2 lies within the boundaries of rect1
    return (x1 <= centroid_x2 <= (x1 + w1)) and (y1 <= centroid_y2 <= (y1 + h1))
    
def convert_to_absolute(coords, frame_shape, factor=1):
    _, x_center, y_center, width, height = map(float, coords)

    width *= factor
    height *= factor

    x_center_abs = int(x_center * frame_shape[1])
    y_center_abs = int(y_center * frame_shape[0])
    width_abs = int(width * frame_shape[1])
    height_abs = int(height * frame_shape[0])

    x1 = int(x_center_abs - (width_abs / 2))
    y1 = int(y_center_abs - (height_abs / 2))

    return x1, y1, width_abs, height_abs

# get label folder
absolute_path = os.path.dirname(os.path.abspath(__file__))
relative_path = 'labels'
label_folder = os.path.join(absolute_path, relative_path)

# get label file
label = [label for label in os.listdir(label_folder)]
label.sort()

# read label file and put it into a list
label_container = []    
for la in label:
    with open(os.path.join(label_folder, la), 'r') as f:
        line = f.read().splitlines()
        line_ac = [float(val) for sublist in line for val in sublist.split()]
        label_container.append(line_ac)
        
# read video file
video = cv2.VideoCapture('video1.mp4')
if not video.isOpened():
    print("Error: Could not open video.")
    exit()
    
# Read first frame
ret, frame = video.read()

# convert label to absolute coordinates
label_container_abs = [convert_to_absolute(coords, frame.shape) for coords in label_container]

# get first label and remove it from the container
first_location = label_container_abs.pop(0)


tracker_type_name = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE', 'DaSiamRPN', 'GOTURN', 'CSRT_NONLEGACY']

tracker = []
for i in range(3):
    tracker.append(create_tracker(i))
    tracker[i].init(frame, first_location)

color = [(0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 0, 0)]

centroid_x_avg, centroid_y_avg, dividor = 0, 0, 0

for i in range(len(label_container_abs)):
    ret, frame = video.read()
    if not ret:
        break
    centroid_x = int(label_container_abs[i][0] + label_container_abs[i][2]/2)
    centroid_y = int(label_container_abs[i][1] + label_container_abs[i][3]/2)
    for j in range(3):
        success, box = tracker[j].update(frame)
        if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x+w, y+h), color[j], 2)
                cv2.putText(frame, f'{tracker_type_name[j]} Tracker', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color[j], thickness=2)
                centroid_x_tracking = int(x + w/2)
                centroid_y_tracking = int(y + h/2)
                cv2.circle(frame, (centroid_x_tracking, centroid_y_tracking), radius=5, color=color[j], thickness=-1)  # The -1 thickness fills the circle
        else:
            tracker[j] = create_tracker(j)
            x_new, y_new, w_new, h_new = label_container_abs[i]
            tracker[j].init(frame, (x_new, y_new, w_new, h_new))
    cv2.imshow('Frame', frame)
    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break
cv2.destroyAllWindows()
