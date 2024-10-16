import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file into a DataFrame
df = pd.read_csv('mahalanobis_distances_by_frame.csv')

# Ensure the Frame column is correctly set
df['Frame'] = range(len(df))

# Create the plot
plt.figure(figsize=(12, 6))

# Define a color map for each tracker to keep the X markers in the same color
color_map = {
    'CSRT': 'orange',
    'KCF': 'red',
    'MIL': 'magenta',
    'BOOSTING': 'blue',
    'MEDIANFLOW': 'cyan',
    'MOSSE': 'green'
}

# Plot each tracker, marking missing values with 'X' in the same color as the tracker
for tracker in ['CSRT', 'KCF', 'MIL', 'BOOSTING', 'MEDIANFLOW', 'MOSSE']:
    # Plot the line for non-missing values
    plt.plot(df['Frame'], df[tracker], label=tracker, color=color_map[tracker])
    
    # Find indices where data is missing (NaN)
    nan_indices = df.index[df[tracker].isna()]
    
    # For each NaN value, place 'X' at the midpoint of previous and next valid data points
    for nan_index in nan_indices:
        # Get previous and next valid values
        if nan_index > 0 and nan_index < len(df) - 1:
            prev_value = df[tracker].iloc[nan_index - 1]
            next_value = df[tracker].iloc[nan_index + 1]
            
            # Calculate the midpoint between previous and next values
            if not np.isnan(prev_value) and not np.isnan(next_value):
                midpoint = (prev_value + next_value) / 2
                plt.scatter(df['Frame'].iloc[nan_index], midpoint, color=color_map[tracker], marker='x', s=100)

# Add titles and labels
plt.title('Mahalanobis Distances for Each Tracker Over Time (with tracker failures marked in the same color)')
plt.xlabel('Frame Number')
plt.ylabel('Mahalanobis Distance')

# Set the x-axis limits based on actual frame numbers
plt.xlim(df['Frame'].min(), df['Frame'].max())

# Add legend and grid
plt.legend(loc='best')
plt.grid(True)

# Show plot with corrected frame number scaling and X markers in the same color
plt.show()
