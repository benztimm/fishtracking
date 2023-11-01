import cv2
import os

import csv
import sys
if len(sys.argv) != 2:
    print("Error: Invalid arguments.")
    print("Usage: python video_count.py tracker_type")
    print("Needed one argument: tracker_type:Number ")
    print("The tracker type should be an integer in the range [0, 6].")
    print("0: BOOSTING, 1: MIL, 2: KCF, 3: TLD, 4: MEDIANFLOW, 5: CSRT, 6: MOSSE")
    exit()

tracker_type = int(sys.argv[1])
if tracker_type < 0 or tracker_type > 6:
    print("Error: Invalid tracker type.")
    print("The tracker type should be an integer in the range [0, 6].")
    print("0: BOOSTING, 1: MIL, 2: KCF, 3: TLD, 4: MEDIANFLOW, 5: CSRT, 6: MOSSE")
    exit()
tracker_type_name = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE', 'DaSiamRPN', 'GOTURN', 'CSRT_NONLEGACY']
absolute_path = os.path.dirname(os.path.abspath(__file__))
relative_path = 'labels'
label_folder = os.path.join(absolute_path, relative_path)

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
        return cv2.legacy.TrackerCSRT.create()
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


def is_overlapping(rect1, rect2):
    x1_1, y1_1, w1, h1 = rect1
    x2_1, y2_1, w2, h2 = rect2

    x1_2, y1_2 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x2_1 + w2, y2_1 + h2

    return not (x1_2 < x2_1 or x1_1 > x2_2 or y1_2 < y2_1 or y1_1 > y2_2)

def is_centroid_overlapping(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Compute the centroid of rect2
    centroid_x2 = x2 + w2 // 2
    centroid_y2 = y2 + h2 // 2

    # Check if the centroid of rect2 lies within the boundaries of rect1
    return (x1 <= centroid_x2 <= (x1 + w1)) and (y1 <= centroid_y2 <= (y1 + h1))



# load labels from file to array
label = [label for label in os.listdir(label_folder)]
label.sort()
label_container = []    
for la in label:
    with open(os.path.join(label_folder, la), 'r') as f:
        line = f.read().splitlines()
        line_ac = [float(val) for sublist in line for val in sublist.split()]
        label_container.append(line_ac)
    
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

# Create a tracker object
tracker = create_tracker(tracker_type)

# Initialize the tracker with the calculated bounding box
tracker.init(frame, first_location) 


# initialize count variables
count_no_overlap = 0
count_tracking_failed = 0
distances = []
# since we have label for every frame, we read video with count of label_container_abs
for i in range(len(label_container_abs)):
    ret, frame = video.read()
    if not ret:
        break
    
    success, box = tracker.update(frame)
    if success:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{tracker_type_name[tracker_type]} Tracker', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        centroid_x_tracking = int(x + w/2)
        centroid_y_tracking = int(y + h/2)
        cv2.circle(frame, (centroid_x_tracking, centroid_y_tracking), radius=5, color=(0, 255, 0), thickness=-1)  # The -1 thickness fills the circle
        
        cv2.rectangle(frame, (label_container_abs[i][0],label_container_abs[i][1]), (label_container_abs[i][0]+label_container_abs[i][2], label_container_abs[i][1]+label_container_abs[i][3]), (255, 0, 0), 2)
        cv2.putText(frame, f'Real Fish Location', (label_container_abs[i][0], label_container_abs[i][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        centroid_x = int(label_container_abs[i][0] + label_container_abs[i][2]/2)
        centroid_y = int(label_container_abs[i][1] + label_container_abs[i][3]/2)
        cv2.circle(frame, (centroid_x, centroid_y), radius=5, color=(255, 0, 0), thickness=-1)  # The -1 thickness fills the circle
        distance = ((centroid_x - centroid_x_tracking)**2 + (centroid_y - centroid_y_tracking)**2)**0.5
        distances.append(distance)        
        if distance > 80:
            count_no_overlap = count_no_overlap + 1
            # Delete the existing tracker and create a new one
            tracker = create_tracker(tracker_type)
            x_new, y_new, w_new, h_new = label_container_abs[i]
            
            # Initialize the tracker with the new bounding box
            tracker.init(frame, (x_new, y_new, w_new, h_new))
    else:
        # Delete the existing tracker and create a new one
        tracker = create_tracker(tracker_type)
        x_new, y_new, w_new, h_new = label_container_abs[i]
        
        # Initialize the tracker with the new bounding box
        tracker.init(frame, (x_new, y_new, w_new, h_new))
        
        # Draw bounding box for the "real" fish location
        cv2.rectangle(frame, (label_container_abs[i][0],label_container_abs[i][1]), (label_container_abs[i][0]+label_container_abs[i][2], label_container_abs[i][1]+label_container_abs[i][3]), (255, 0, 0), 2)
        cv2.putText(frame, f'Real Fish Location', (label_container_abs[i][0], label_container_abs[i][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        centroid_x = int(label_container_abs[i][0] + label_container_abs[i][2]/2)
        centroid_y = int(label_container_abs[i][1] + label_container_abs[i][3]/2)
        cv2.circle(frame, (centroid_x, centroid_y), radius=5, color=(255, 0, 0), thickness=-1)  # The -1 thickness fills the circle
        count_tracking_failed = count_tracking_failed + 1
        distances.append(1000)  

    
    cv2.putText(frame, f'No Overlap Count: {count_no_overlap}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Tracking Failed Count {count_tracking_failed}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f'Tracker Type: {tracker_type_name[tracker_type]}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    #macbook pro resolution
    #frame_resized = cv2.resize(frame, (2576,1610))
    
    #pc resolution
    frame_resized = cv2.resize(frame, (1600,900))

    cv2.imshow(tracker_type_name[tracker_type], frame_resized)
    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break
cv2.destroyAllWindows()



print(f'No Overlap Count: {count_no_overlap}')
print(f'Tracking Failed Count {count_tracking_failed}')

def write_tracker(filename, tracker_type, time_overlapped, time_tracker_failed,framecount):
    
    template = f"""
time overlapped: {time_overlapped} %Time Overlapped = {time_overlapped/framecount*100}%
time tracker failed: {time_tracker_failed} %Time Tracker Failed = {time_tracker_failed/framecount*100}%
tracker type: {tracker_type}
framecount : {framecount}
"""

    # Check and write if the tracker details don't already exist
    write_if_not_exists(filename, template)

def write_if_not_exists(filename, text):
    # Create the file if it doesn't exist, or open it for reading
    with open(filename, 'a+') as f:
        f.seek(0)  # move file cursor to the beginning of the file for reading
        existing_content = f.read()
        if text in existing_content:
            print(f"The text for tracker already exists in {filename}. Skipping.")
            return
        
        # If the text doesn't exist in the file, append it
        f.write(text)
        print(f"Tracker details written to {filename}.")

if count_no_overlap or count_tracking_failed:
    filename = "trackers_distance.txt"
    framecount = len(label_container_abs)+1
    write_tracker(filename, tracker_type_name[tracker_type], count_no_overlap, count_tracking_failed,framecount)

if len(label_container_abs) == len(distances):
    with open(f'distances_dist_{tracker_type_name[tracker_type]}.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Frame Number', 'Distance'])  # Header row
        for idx, distance in enumerate(distances, start=1):  
            csvwriter.writerow([idx, distance])
