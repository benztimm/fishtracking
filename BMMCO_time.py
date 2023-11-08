"""

E. Wes Bethel, Copyright (C) 2022

October 2022

Description: This code loads a .csv file and creates a 3-variable plot, and saves it to a file named "myplot.png"

Inputs: the named file "sample_data_3vars.csv"

Outputs: displays a chart with matplotlib

Dependencies: matplotlib, pandas modules

Assumptions: developed and tested using Python version 3.8.8 on macOS 11.6

"""

import pandas as pd
import matplotlib.pyplot as plt

plot_fname = "elapsedtime-combine.png"
fname = "bmmco_time.csv"
df = pd.read_csv(fname, comment="#")
print(df)
var_names = list(df.columns)
print("var names =", var_names)

# split the df into individual vars
# assumption: column order - 0=problem size, 1=blas time, 2=basic time

problem_sizes = df[var_names[0]].values.tolist()
Box = df[var_names[1]].values.tolist()
Centroid = df[var_names[2]].values.tolist()
Distance = df[var_names[3]].values.tolist()

plt.figure()
plt.title("Time comparison for each tracker method for each tracker type")
xlocs = [i for i in range(len(problem_sizes))]

plt.xticks(xlocs, problem_sizes)

plt.plot(Box, "r-o",label = var_names[1])
plt.plot(Centroid, "b-x",label = var_names[2])
plt.plot(Distance, "g-^",label = var_names[3])


plt.legend(loc="best")
plt.grid(axis='both')

plt.xlabel("Tracker Type")
plt.ylabel("Time (seconds)")



plt.savefig(plot_fname, dpi=300)

plt.show()

# EOF