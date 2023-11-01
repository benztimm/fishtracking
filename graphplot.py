import pandas as pd
import matplotlib.pyplot as plt
import math


import sys

if len(sys.argv) != 2:
    print("Error: Invalid arguments.")
    print("Usage: python graphplot.py filename")
    print("Needed one argument: filename:Number ")
    print("The filename argument should be an integer in the range [0, 2].")
    print("0: distances_combine.csv, 1: distances_combine_box.csv, 2: distances_combine_dist.csv")
    exit()

fnum = int(sys.argv[1])
if fnum < 0 or fnum > 2:
    print("Error: Invalid tracker type.")
    print("The filename argument should be an integer in the range [0, 2].")
    print("0: distances_combine.csv, 1: distances_combine_box.csv, 2: distances_combine_dist.csv")
    exit()

fnames = ["distances_combine.csv","distances_combine_box.csv","distances_combine_dist.csv"]
output_names = ["distances_combine_centroid.png","distances_combine_box.png","distances_combine_dist.png"]
titles = ["Distance vs Frame Number (Centroid)","Distance vs Frame Number Box check (Combine)","Distance vs Frame Number (Distance)"]
# Reading the data
fname = fnames[fnum]
filename = output_names[fnum]
df = pd.read_csv(fname, comment="#")
var_names = list(df.columns)

# Define a function to replace 1000 with NaN (Not a Number) 
# so that it won't get plotted on the line plot
def replace_failures(data):
    return [value if value != 1000 else None for value in data]

# Create a function to get failure points (where value is 1000)
def get_failures(data):
    return [i for i, value in enumerate(data) if value == 1000]

# Color dictionary for each tracker's failure markers
failure_colors = {
    var_names[1]: "#C0392B",  # Dark Red
    var_names[2]: "#9B59B6",  # Purple
    var_names[3]: "#2980B9",  # Blue
    var_names[4]: "#1ABC9C",  # Turquoise
    var_names[5]: "#DFFF00",  # Yellow
    var_names[6]: "#808080",  # Gray
    var_names[7]: "#17202A"  # Dark Gray
}

# Plotting
plt.figure(figsize=(18, 8))
plt.title(titles[fnum])

# For each tracking method, plot the distances and failures
for idx, method in enumerate(var_names[1:], start=1):
    data = df[method].values.tolist()
    
    # Get cleaned data and failure points
    cleaned_data = replace_failures(data)
    failure_points = get_failures(data)
    print(max([x for x in cleaned_data if x is not None]))
    # Plot the cleaned data with the color from the failure_colors dictionary
    plt.plot(cleaned_data, label=method, color=failure_colors[method], alpha=1)
    
    # Plot the failures using the same color with increased marker size
    for point in failure_points:
        plt.plot(point, max([x for x in cleaned_data if x is not None]), 'x', color=failure_colors[method], markersize=10, alpha=1)
    
    # Calculate and plot the maximum line for each tracker
    max_value = max([x for x in cleaned_data if x is not None])
    plt.axhline(y=max_value, color=failure_colors[method], linestyle='--', alpha=0.5)

plt.legend(loc="best")
plt.grid(axis='both')
plt.ylim(0, 1000)


plt.savefig(filename, dpi=300)
plt.show()
