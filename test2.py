import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# State vector
statePost = np.array([527.83624, 486.19244, -5.202041, -3.2854788])

# Extract position and velocity
position = statePost[:2]  # [x, y]
velocity = statePost[2:]  # [vx, vy]

# Example covariance matrix for position (replace with actual covariance)
position_covariance = np.array([[0.5469229, 0],
                                [0., 0.5469229]])

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(position_covariance)

# Compute the lengths of the ellipse axes (2*sqrt(eigenvalue) for 95% confidence interval)
axis_lengths = 2 * np.sqrt(eigenvalues)

# Compute the angle of the ellipse in degrees
angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

# Define the center point of the ellipse
center = (position[0], position[1])

# Plotting the position, ellipse, and direction
fig, ax = plt.subplots()

# Plot the position as a blue dot
plt.plot(position[0], position[1], 'bo')

# Plot the ellipse representing the uncertainty
ellipse = Ellipse(xy=center, width=axis_lengths[0], height=axis_lengths[1], angle=angle, edgecolor='r', fc='None', lw=2)
ax.add_patch(ellipse)

# Scale the velocity for better visualization
scale = 10
end_pt = (position[0] + scale * velocity[0], position[1] + scale * velocity[1])
plt.arrow(position[0], position[1], scale * velocity[0], scale * velocity[1], head_width=10, head_length=20, fc='g', ec='g')

plt.xlim(400, 600)
plt.ylim(400, 600)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Position, Uncertainty Ellipse, and Direction')
plt.grid()
plt.show()
