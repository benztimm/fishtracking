import matplotlib.pyplot as plt
import csv
import sys

if len(sys.argv) != 2:
    print("Error: Invalid arguments.")
    print("Usage: python plot.py tracker_type")
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
# Lists to store the data
frame_numbers = []
distances = []

# Read the CSV file
with open(f'distances_{tracker_type_name[tracker_type]}.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip the header row
    for row in csvreader:
        frame_numbers.append(int(row[0]))
        distances.append(float(row[1]))

# Plot the entire data first
plt.figure(figsize=(10, 6))
plt.plot(frame_numbers, distances, marker='o', linestyle='-')
plt.xlabel('Frame Number')
plt.ylabel('Distance (pixels)')
plt.title(f'Distance vs Frame Number ({tracker_type_name[tracker_type]})')
plt.grid(True)
plt.tight_layout()

filename = f'distance_plot_{tracker_type_name[tracker_type]}.png'
plt.savefig(filename)

plt.show()


