import cv2
import os
import numpy as np
import csv
import sys
import time

TRACKER_NUM = 1
tracker_type_name = ['MOSSE']

def create_tracker(tracker_type):
    if tracker_type == 0:
        return cv2.legacy.TrackerMOSSE.create()

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

def cluster_centroids(centroids, min_distance=20):
    clusters = [[centroids[0]]]  # Start with the first centroid in its own cluster

    # Attempt to place each centroid into a cluster based on the distance to the first item in each cluster
    for centroid in centroids[1:]:
        add_to_cluster = True
        for existing_cluster in clusters:
            if calculate_distance(existing_cluster[0], centroid) < min_distance:
                existing_cluster.append(centroid)
                add_to_cluster = False
                break
        if add_to_cluster:
            clusters.append([centroid])
            
        for cluster in clusters:
            cluster.sort()
    return clusters


def calculate_combined_average(clusters):
    total_trackers = sum(len(cluster) for cluster in clusters)
    if len(clusters) == total_trackers:
        # Edge case: Each tracker in its own list, calculate overall average
        all_trackers = [tracker for cluster in clusters for tracker in cluster]
        avg_x = sum(x for x, _ in all_trackers) // len(all_trackers)
        avg_y = sum(y for _, y in all_trackers) // len(all_trackers)
        return (avg_x, avg_y)
    else:
        # Normal case: Find the largest cluster and calculate its average
        largest_cluster = max(clusters, key=len)
        avg_x = sum(x for x, _ in largest_cluster) // len(largest_cluster)
        avg_y = sum(y for _, y in largest_cluster) // len(largest_cluster)
        return (avg_x, avg_y)


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
    

tracker = create_tracker(0)
tracker.init(frame, first_location)

color = [(255, 0, 0),(0, 255, 0),(0, 0, 255),(255, 255, 0),(255, 0, 255),(0, 255, 255),(128, 0, 128),(255, 165, 0)]

for i in range(len(label_container_abs)):
    ret, frame = video.read()
    if not ret:
        break
    centroid_x = int(label_container_abs[i][0] + label_container_abs[i][2]/2)
    centroid_y = int(label_container_abs[i][1] + label_container_abs[i][3]/2)
    #cv2.putText(frame, f'{tracker_type_name[j]} Tracker', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255,255,255), thickness=2)
    cv2.rectangle(frame, (label_container_abs[i][0],label_container_abs[i][1]), (label_container_abs[i][0]+label_container_abs[i][2], label_container_abs[i][1]+label_container_abs[i][3]), color=(0,0,0), thickness=2)
    cv2.circle(frame, (centroid_x, centroid_y), radius=5, color=(0,0,0), thickness=-1)  # The -1 thickness fills the circle
    cv2.putText(frame, f'Real Fish Location', (centroid_x, centroid_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,0,0), thickness=2)
    
    success, box = tracker.update(frame)
    if success:
        (x, y, w, h) = [int(v) for v in box]
 
        centroid_x_tracking = int(x + w/2)
        centroid_y_tracking = int(y + h/2)
        
        cv2.rectangle(frame, (x,y), (x+w, y+h), color=(255,255,255), thickness=2)
        cv2.circle(frame, (centroid_x_tracking, centroid_y_tracking), radius=5, color=(255,255,255), thickness=-1)  # The -1 thickness fills the circle
        cv2.putText(frame, f'{tracker_type_name[0]}', (centroid_x_tracking, centroid_y_tracking-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255,255,255), thickness=2)
        
        if not is_centroid_overlapping((x, y, w, h), label_container_abs[i]):
            # Delete the existing tracker and create a new one
            tracker = create_tracker(0)
            x_new, y_new, w_new, h_new = label_container_abs[i]
            # Initialize the tracker with the new bounding box
            tracker.init(frame, (x_new, y_new, w_new, h_new))
            print(f"Tracker {tracker_type_name[0]} not overlapping with ground truth, reinitializing tracker...")
            
    else:
        print(f"Tracker {tracker_type_name[0]} failed")
        tracker = create_tracker(0)
        x_new, y_new, w_new, h_new = label_container_abs[i]
        tracker.init(frame, (x_new, y_new, w_new, h_new))
    
    cv2.imshow('Frame', frame)
    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break
cv2.destroyAllWindows()


