import cv2
import os
import numpy as np
import csv
import sys
import time
from scipy.optimize import minimize

def multi_covariance_intersection(estimates, covariances):
    num_estimates = len(estimates)
    
    def objective(weights):
        weights = np.clip(weights, 0, 1)
        weights /= np.sum(weights)  # Normalize weights
        weighted_inv_cov_sum = sum(w * np.linalg.inv(P) for w, P in zip(weights, covariances))
        return np.trace(np.linalg.inv(weighted_inv_cov_sum))
    
    initial_weights = np.ones(num_estimates) / num_estimates
    bounds = [(0, 1)] * num_estimates
    result = minimize(objective, initial_weights, bounds=bounds)
    optimal_weights = result.x / np.sum(result.x)
    
    fused_cov = np.linalg.inv(sum(optimal_weights[j] * np.linalg.inv(covariances[j]) for j in range(num_estimates)))
    fused_estimate = fused_cov.dot(sum(optimal_weights[j] * np.linalg.inv(covariances[j]).dot(estimates[j]) for j in range(num_estimates)))
    
    return fused_estimate, fused_cov

TRACKER_NUM = 6
tracker_type_name = ['CSRT', 'KCF', 'MIL', 'BOOSTING', 'MEDIANFLOW',  'MOSSE', 'TLD']

def create_tracker(tracker_type):
    if tracker_type == 0:
        return cv2.TrackerCSRT.create()
    if tracker_type == 1:
        return cv2.legacy.TrackerKCF.create()
    if tracker_type == 2:
        return cv2.legacy.TrackerMIL.create()
    if tracker_type == 3:
        return cv2.legacy.TrackerBoosting.create()
    if tracker_type == 4:
        return cv2.legacy.TrackerMedianFlow.create()
    if tracker_type == 5:
        return cv2.legacy.TrackerMOSSE.create()
    if tracker_type == 6:
        return cv2.legacy.TrackerTLD.create()

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
    
    
def calculate_distance(centroid1, centroid2):
    return np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)


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
    
frame_container = []    
# Read first frame
ret, frame = video.read()
frame_container.append(frame)
# convert label to absolute coordinates
label_container_abs = [convert_to_absolute(coords, frame.shape) for coords in label_container]

# get first label and remove it from the container
first_location = label_container_abs.pop(0)

tracker = []
kalman = []
position_estimates = []
position_covariances = []

for i in range(TRACKER_NUM):
    tracker.append(create_tracker(i))
    tracker[i].init(frame, first_location)
    kalman.append(cv2.KalmanFilter(4, 2))  # 4 dynamic params (x, y, dx, dy), 2 measurement params (x, y)
    kalman[i].measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman[i].transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman[i].processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 5, 0], [0, 0, 0, 5]], np.float32) * 0.03
    kalman[i].measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1

color = [(255, 0, 0),(0, 255, 0),(0, 0, 255),(255, 255, 0),(255, 0, 255),(0, 255, 255),(128, 0, 128),(255, 165, 0)]




for i in range(len(label_container_abs)):
    ret, frame = video.read()
    if not ret:
        break
    for j in range(TRACKER_NUM):
        success, box = tracker[j].update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            centroid_x_tracking = int(x + w/2)
            centroid_y_tracking = int(y + h/2)
            
            #cv2.rectangle(frame, (x,y), (x+w, y+h), color=color[j], thickness=2)
            cv2.circle(frame, (centroid_x_tracking, centroid_y_tracking), radius=5, color=color[j], thickness=-1)  # The -1 thickness fills the circle
            #cv2.putText(frame, f'{tracker_type_name[j]}', (centroid_x_tracking, centroid_y_tracking-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color[j], thickness=2)
            
            if not is_centroid_overlapping((x, y, w, h), label_container_abs[i]):
                # Delete the existing tracker and create a new one
                tracker[j] = create_tracker(j)
                x_new, y_new, w_new, h_new = label_container_abs[i]
                # Initialize the tracker with the new bounding box
                tracker[j].init(frame, (x_new, y_new, w_new, h_new))
                print(f"Tracker {tracker_type_name[j]} not overlapping with ground truth, reinitializing tracker...")
                
             # Update Kalman Filter with the new measurement
            measurement = np.array([[np.float32(centroid_x_tracking)], [np.float32(centroid_y_tracking)]])
            kalman[j].correct(measurement)
            # Get the new prediction
            prediction = kalman[j].predict()
            # Draw the predicted position
            pt = (int(prediction[0]), int(prediction[1]))
            #cv2.circle(frame, pt, radius=5, color=color[j], thickness=-1)
            #cv2.putText(frame, f'kalman_{tracker_type_name[j]}', (int(prediction[0]), int(prediction[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color[j], thickness=2)
            state_estimate = kalman[j].statePost
            covariance_matrix = kalman[j].errorCovPost
            # If needed, print or store these values
            position_estimate = kalman[j].statePost[:2]
            position_covariance = kalman[j].errorCovPost[:2, :2]

            position_estimates.append(position_estimate)
            position_covariances.append(position_covariance)
        else:
            print(f"Tracker {tracker_type_name[j]} failed")
            tracker[j] = create_tracker(j)
            x_new, y_new, w_new, h_new = label_container_abs[i]
            tracker[j].init(frame, (x_new, y_new, w_new, h_new))
            
    # plot ground truth
    centroid_x = int(label_container_abs[i][0] + label_container_abs[i][2]/2)
    centroid_y = int(label_container_abs[i][1] + label_container_abs[i][3]/2)
    cv2.putText(frame, f'Real Fish Location', (label_container_abs[i][0], label_container_abs[i][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,0,0), thickness=2)
    cv2.rectangle(frame, (label_container_abs[i][0],label_container_abs[i][1]), (label_container_abs[i][0]+label_container_abs[i][2], label_container_abs[i][1]+label_container_abs[i][3]), color=(0,0,0), thickness=2)
    cv2.circle(frame, (centroid_x, centroid_y), radius=7, color=(0,0,0), thickness=-1)  # The -1 thickness fills the circle
    #cv2.putText(frame, f'Real Fish Centroid', (centroid_x, centroid_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,0,0), thickness=2)
    
    if position_estimates:
        fused_position, fused_covariance = multi_covariance_intersection(position_estimates, position_covariances)

        # Draw the fused position as a circle and the uncertainty as an ellipse
        x, y = int(fused_position[0]), int(fused_position[1])
        cv2.circle(frame, (x, y), 7, (255, 255, 255), -1)
        cv2.putText(frame, f'Fused location', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255,255,255), thickness=2)

    frame_container.append(frame)
    position_covariances.clear()
    position_estimates.clear()
    cv2.imshow('Frame', frame)
    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break
cv2.destroyAllWindows()


height, width = frame_container[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
fps = 10.0  # Frames per second
out = cv2.VideoWriter('fused_output2.mp4', fourcc, fps, (width, height))
for frame in frame_container:
    out.write(frame)
out.release()

