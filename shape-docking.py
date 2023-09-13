import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
from skimage.morphology import local_minima
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

def compute_histograms(polygon, reverse=False):
    def angle_between_lines(p1, p2, p3):
        """p1: (x1, y1), p2: (x2, y2), p3: (x3, y3)
        Returns the angle between the lines p1p2 and p2p3 in degrees"""
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        v1 = p1 - p2
        v2 = p3 - p2
        angle = np.degrees(np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2)))
        if angle < 0:  # Convert negative angles to their positive equivalent
            angle += 360
        return angle

    n = len(polygon)
    angles = []
    side_lengths = []
    normals = []

    for i in range(n):
        if i == 0:
            prev_point = polygon[-1]
        else:
            prev_point = polygon[i-1]
        if i == n-1:
            next_point = polygon[0]
        else:
            next_point = polygon[i+1]

        # Get three consecutive points
        p1, p2, p3 = prev_point, polygon[i], next_point

        # Calculate the internal angle at p2
        angle = angle_between_lines(p1, p2, p3)

        angles.append(angle)

        # Calculate side length
        side = np.array(p3) - np.array(p2)
        length = np.linalg.norm(side)
        side_lengths.append(length)

        # Calculate normal orientation (assuming it points to the left of the direction of travel)
        normal = np.array([-side[1], side[0]])
        normal_orientation = np.degrees(np.arctan2(normal[1], normal[0]))
        normals.append(normal_orientation)
    
    if reverse:
        angles = [360 - angle for angle in angles]
        normals = [normal*-1 for normal in normals]

    return angles, side_lengths, normals

def plot_histograms(polygon, ax0, ax1, ax2, reverse=False, highlight_side=None):
    angles, side_lengths, normals = compute_histograms(polygon, reverse=reverse)
    
    # Plot the polygon itself
    x, y = zip(*polygon + [polygon[0]])  # Adding the first point to close the polygon
    for i, (x1, y1, x2, y2) in enumerate(zip(x[:-1], y[:-1], x[1:], y[1:])):
        if i == highlight_side:
            ax0.plot([x1, x2], [y1, y2], '-o', linewidth=2.5, markersize=7, color='red')
        else:
            ax0.plot([x1, x2], [y1, y2], '-o', linewidth=1.5, markersize=5, color='C0')
        # add side numbers to each side near the middle
        ax0.text((x1 + x2)/2, (y1 + y2)/2, str(i+1), ha="center", va="center", fontsize=8)
    ax0.set_title("Polygon")
    ax0.set_aspect('equal', 'box')

    side_lengths.append(side_lengths[-1])
    angles.append(angles[-1])
    normals.append(normals[-1])

    edges = np.cumsum(side_lengths)  # Add a zero to the end to make the lengths and angles the same length
    # Angle Histogram
    ax1.fill_between(edges, angles, step="post", edgecolor='C1', facecolor='none', linewidth=1.5)
    # Add numbers to the top of each bar in the middle of the bin
    for i, (x, y) in enumerate(zip(edges, angles)):
        if i < len(edges) - 1:
            # small font size to fit the numbers in the bins
            ax1.text(x + side_lengths[i]/2, y, str(i+1), ha="center", va="bottom", fontsize=6)
            # show small certical lines to indicate the bin edges
            ax1.plot([x, x], [0, y], color='C1', linewidth=0.2)

    
    # add vertical lines to indicate the bin edges
    ax1.set_xticks(edges)
    ax1.set_xticklabels([str(i+1) for i in range(len(edges))])
    ax1.set_title("Angle Histogram")
    ax1.set_xlabel("Cumulative Length")
    ax1.set_ylabel("Angle (degrees)")

    # Normal Orientation Histogram
    ax2.fill_between(edges, normals, step="post", edgecolor='C2', facecolor='none', linewidth=1.5)
    # Add numbers to the top of each bar in the middle of the bin
    for i, (x, y) in enumerate(zip(edges, normals)):
        if i < len(edges) - 1:
            # small font size to fit the numbers in the bins
            ax2.text(x + side_lengths[i]/2, y, str(i+1), ha="center", va="bottom", fontsize=6)
            # show small certical lines to indicate the bin edges
            ax2.plot([x, x], [0, y], color='C2', linewidth=0.2)
    
    # add vertical lines to indicate the bin edges
    ax2.set_xticks(edges)
    ax2.set_xticklabels([str(i+1) for i in range(len(edges))])
    ax2.set_title("Normal Orientation Histogram")
    ax2.set_xlabel("Cumulative Length")
    ax2.set_ylabel("Normal Orientation (degrees)")

def compute_all_differences(polygon1, polygon2):
    angles1, _, normals1 = compute_histograms(polygon1)
    angles2, _, normals2 = compute_histograms(polygon2, reverse=True)
    
    angle_diff_matrix = []
    normal_diff_matrix = []

    # Compute the difference between each angle in polygon1 and polygon2
    for angle2 in angles2:
        row = []
        for angle1 in angles1:
            diff = abs(angle2 - angle1)
            row.append(diff)
        angle_diff_matrix.append(row)
    
    # Compute the difference between each normal orientation in polygon1 and polygon2
    for normal2 in normals2:
        row = []
        for normal1 in normals1:
            diff = abs(normal2 - normal1)
            row.append(diff)
        normal_diff_matrix.append(row)

    return np.array(angle_diff_matrix), np.array(normal_diff_matrix)

def plot_3d_difference(polygon1, polygon2):
    angle_diff_matrix, normal_diff_matrix = compute_all_differences(polygon1, polygon2)

    x = np.arange(1, len(polygon1) + 1)
    y = np.arange(1, len(polygon2) + 1)
    x, y = np.meshgrid(x, y)
    
    fig, ax = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    
    ax[0].plot_surface(x, y, angle_diff_matrix, cmap='viridis', linewidth=0.2)
    ax[1].plot_surface(x, y, normal_diff_matrix, cmap='viridis', linewidth=0.2)
    
    ax[0].set_xlabel('Polygon1 Edges')
    ax[0].set_ylabel('Polygon2 Edges')
    ax[0].set_zlabel('Angle Difference')
    ax[0].set_title("Angle Difference between Polygon Edges")

    ax[1].set_xlabel('Polygon1 Edges')
    ax[1].set_ylabel('Polygon2 Edges')
    ax[1].set_zlabel('Normal Orientation Difference')
    ax[1].set_title("Normal Orientation Difference between Polygon Edges")

    # Adjusting x-axis tick labels to start from 1
    ax[0].set_xticks(np.arange(len(polygon1)))
    ax[0].set_xticklabels([str(i+1) for i in range(len(polygon1))])
    ax[0].set_yticks(np.arange(len(polygon2)))
    ax[0].set_yticklabels([str(i+1) for i in range(len(polygon2))])
    ax[0].invert_yaxis()  # To show the 1st edge of Polygon2 at the top

    # Adjusting x-axis tick labels to start from 1
    ax[1].set_xticks(np.arange(len(polygon1)))
    ax[1].set_xticklabels([str(i+1) for i in range(len(polygon1))])
    ax[1].set_yticks(np.arange(len(polygon2)))
    ax[1].set_yticklabels([str(i+1) for i in range(len(polygon2))])
    ax[1].invert_yaxis()  # To show the 1st edge of Polygon2 at the top

    plt.show()

def plot_heatmap_difference(polygon1, polygon2):
    angle_diff_matrix, normal_diff_matrix = compute_all_differences(polygon1, polygon2)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    sns.heatmap(angle_diff_matrix, cmap='viridis', ax=ax[0], cbar=False, annot=True, fmt=".0f")
    ax[0].set_xlabel('Polygon1 Edges')
    ax[0].set_ylabel('Polygon2 Edges')
    ax[0].set_title("Angle Difference between Polygon Edges")
    
    # Adjusting x-axis tick labels to start from 1
    ax[0].set_xticks(np.arange(len(polygon1)))
    ax[0].set_xticklabels([str(i+1) for i in range(len(polygon1))])
    ax[0].set_yticks(np.arange(len(polygon2)))
    ax[0].set_yticklabels([str(i+1) for i in range(len(polygon2))])
    ax[0].invert_yaxis()  # To show the 1st edge of Polygon2 at the top
    
    sns.heatmap(normal_diff_matrix, cmap='viridis', ax=ax[1], cbar=False, annot=True, fmt=".0f")
    ax[1].set_xlabel('Polygon1 Edges')
    ax[1].set_ylabel('Polygon2 Edges')
    ax[1].set_title("Normal Orientation Difference between Polygon Edges")

    # Adjusting x-axis tick labels to start from 1
    ax[1].set_xticks(np.arange(len(polygon1)))
    ax[1].set_xticklabels([str(i+1) for i in range(len(polygon1))])
    ax[1].set_yticks(np.arange(len(polygon2)))
    ax[1].set_yticklabels([str(i+1) for i in range(len(polygon2))])
    ax[1].invert_yaxis()  # To show the 1st edge of Polygon2 at the top

    # Detect local minima in angle_diff_matrix
    angle_mins = local_minima(angle_diff_matrix)
    angle_pairs = list(zip(*np.where(angle_mins)))
    normal_diff_values = normal_diff_matrix[angle_mins]
    # Plotting red dots on minima locations for both heatmaps based on angle_minima locations
    for (y, x) in angle_pairs:
        ax[0].scatter(x + 0.5, y + 0.5, color='red', s=100)  # +0.5 to center dot in the square
        ax[1].scatter(x + 0.5, y + 0.5, color='red', s=100)  # +0.5 to center dot in the square

    plt.tight_layout()
    plt.show()

    return angle_pairs, normal_diff_values

def midpoint(point1, point2):
    return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

def is_similar(polygon1, polygon2):
    """Check if two polygons are similar based on their area and centroid."""
    return abs(polygon1.area - polygon2.area) < 1e-6 and \
           abs(polygon1.centroid.x - polygon2.centroid.x) < 1e-6 and \
           abs(polygon1.centroid.y - polygon2.centroid.y) < 1e-6

def plot_docked_polygons(polygon1, polygon2, angle_minima_pairs, normal_diff, show_all=False):
    valid_configs = []
    seen_configs = []
    
    for ((y, x), rotation) in zip(angle_minima_pairs, normal_diff):
        midpoint1 = midpoint(polygon1[x], polygon1[(x + 1) % len(polygon1)])
        midpoint2 = midpoint(polygon2[y], polygon2[(y + 1) % len(polygon2)])
        trans_x, trans_y = midpoint1[0] - midpoint2[0], midpoint1[1] - midpoint2[1]
        translated_polygon2 = translate(Polygon(polygon2), xoff=trans_x, yoff=trans_y)
        rotated_polygon2 = rotate(translated_polygon2, rotation, origin=midpoint1)
        
        if not Polygon(polygon1).intersects(rotated_polygon2.buffer(-0.01)) or show_all:
            is_duplicate = False
            for seen_polygon in seen_configs:
                if is_similar(rotated_polygon2, seen_polygon):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                valid_configs.append(((y, x), rotation))
                seen_configs.append(rotated_polygon2)
    
    n = len(valid_configs)
    
    if n == 0:
        print("No valid configurations found.")
        return

    columns = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / columns))
    fig, axs = plt.subplots(rows, columns, constrained_layout=True)
    if rows == 1 and columns == 1:
        axs = np.array([axs])

    for idx in range(n):
        if rows == 1 or columns == 1:
            ax = axs[idx]
        else:
            ax = axs[idx // columns, idx % columns]

        ((y, x), rotation) = valid_configs[idx]
        
        midpoint1 = midpoint(polygon1[x], polygon1[(x + 1) % len(polygon1)])
        midpoint2 = midpoint(polygon2[y], polygon2[(y + 1) % len(polygon2)])
        trans_x, trans_y = midpoint1[0] - midpoint2[0], midpoint1[1] - midpoint2[1]
        translated_polygon2 = translate(Polygon(polygon2), xoff=trans_x, yoff=trans_y)
        rotated_polygon2 = rotate(translated_polygon2, rotation, origin=midpoint1)
        
        x1, y1 = Polygon(polygon1).exterior.xy
        ax.fill(x1, y1, 'C0', alpha=0.7)
        
        x2, y2 = rotated_polygon2.exterior.xy
        ax.fill(x2, y2, 'C1', alpha=0.7)
        
        ax.set_aspect('equal', 'box')
        ax.set_title("Configuration {}".format(idx+1))
        # hide bouding box
        ax.axis('off')

        edge2_coords = list(rotated_polygon2.exterior.coords)
        edge2_x = [edge2_coords[y][0], edge2_coords[(y + 1) % len(edge2_coords)][0]]
        edge2_y = [edge2_coords[y][1], edge2_coords[(y + 1) % len(edge2_coords)][1]]

        ax.plot(edge2_x, edge2_y, 'r-', lw=1)  # Black line for edge2

    # Hide any unused subplots
    for idx in range(n, rows*columns):
        if rows == 1 or columns == 1:
            axs[idx].axis('off')
        else:
            axs[idx // columns, idx % columns].axis('off')

    plt.show()

polygon1 = [(1,3), (2,3), (2,2), (3,2), (3,1), (2,1), (2,0), (1,0), (1,1), (0,1), (0,2), (1,2)]
polygon2 = [(7,3), (8,3), (8,2), (9,2),(9,3), (10,3), (10,1), (7,1)]

fig, axs = plt.subplots(2, 3)

# Plot Histograms for each shape
plot_histograms(polygon1, axs[0, 0], axs[0, 1], axs[0, 2])
plot_histograms(polygon2, axs[1, 0], axs[1, 1], axs[1, 2] , reverse=True)

plt.tight_layout()
plt.show()

# Plot the difference between the two shapes angles as a heatmap
angle_minima_pairs, normal_diff = plot_heatmap_difference(polygon1, polygon2)
print("Minima pairs for Angle Differences:", angle_minima_pairs)
print("Angle to rotate counter clock-wise:", normal_diff)
plot_docked_polygons(polygon1, polygon2, angle_minima_pairs, normal_diff, show_all=False)


