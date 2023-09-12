import numpy as np
from skimage.morphology import local_minima
import matplotlib.pyplot as plt

# Example heatmap (replace with your data)
heatmap = np.array([
    [3, 2, 3, 4],
    [4, 1, 4, 5],
    [5, 3, 0.5, 6],
    [6, 5, 4, 7]
])

# Detect local minima
minima_coordinates = np.argwhere(local_minima(heatmap))

# Extract values of minima
minima_values = heatmap[minima_coordinates[:,0], minima_coordinates[:,1]]

# Plot the heatmap
plt.imshow(heatmap, cmap='viridis', origin='lower')
plt.colorbar(label='Value')

# Mark the minima
plt.scatter(minima_coordinates[:,1], minima_coordinates[:,0], color='red', s=100, marker='o', edgecolors='black')
plt.title('Heatmap with Local Minima')

# Display the plot
plt.show()
