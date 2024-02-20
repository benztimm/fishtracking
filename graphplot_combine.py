import pandas as pd
import matplotlib.pyplot as plt

fnames = ["distances_combine.csv", "distances_combine_box.csv", "distances_combine_dist.csv"]
output_name = "combined_output.png"
titles = ["Distance vs Frame Number (Centroid)", "Distance vs Frame Number Box check (Combine)", "Distance vs Frame Number (Distance)"]

# Function to replace 1000 with NaN
def replace_failures(data):
    return [value if value != 1000 else None for value in data]

# Function to get failure points
def get_failures(data):
    return [i for i, value in enumerate(data) if value == 1000]

# Setup plot - Adjust the figsize according to your screen/display settings
fig, axs = plt.subplots(len(fnames), 1, figsize=(18, 24))

for idx, fname in enumerate(fnames):
    df = pd.read_csv(fname, comment="#")
    var_names = list(df.columns)

    # Custom color dictionary
    failure_colors = {
        var_names[1]: "#C0392B",  # Dark Red
        var_names[2]: "#9B59B6",  # Purple
        var_names[3]: "#2980B9",  # Blue
        var_names[4]: "#1ABC9C",  # Turquoise
        var_names[5]: "#DFFF00",  # Yellow
        var_names[6]: "#808080",  # Gray
        var_names[7]: "#17202A"  # Dark Gray
    }

    axs[idx].set_title(titles[idx])
    axs[idx].set_ylim(0, 1000)
    axs[idx].grid(axis='both')

    for method in var_names[1:]:
        data = df[method].values.tolist()
        cleaned_data = replace_failures(data)
        failure_points = get_failures(data)
        color = failure_colors[method]

        # Plot the data and failures
        axs[idx].plot(cleaned_data, label=method, color=color, alpha=0.7)
        for point in failure_points:
            axs[idx].plot(point, max([x for x in cleaned_data if x is not None]), 'x', color=color, markersize=10, alpha=1)
        
        # Draw the maximum value line
        max_value = max([x for x in cleaned_data if x is not None])
        axs[idx].axhline(y=max_value, color=color, linestyle='--', alpha=0.5)

    axs[idx].legend(loc="best")

plt.tight_layout()
plt.savefig(output_name, dpi=300)
plt.show()
