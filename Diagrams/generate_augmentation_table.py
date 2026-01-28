"""Generate Data Augmentation Table

Creates a simple table showing all data augmentation techniques and their parameters.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Augmentation data
augmentations = [
    ['RandomResizedCrop', 'Scale: (0.9, 1.0), Ratio: (0.95, 1.05)', 'Random crop and resize within 90-100% of original size'],
    ['RandomHorizontalFlip', 'Probability: 0.5', 'Horizontal flip with 50% probability'],
    ['RandomRotation', 'Degrees: ±10°, Probability: 0.3', 'Random rotation up to 10 degrees with 30% probability'],
    ['ColorJitter', 'Brightness: 0.1, Contrast: 0.1, Saturation: 0.1, Hue: 0.02', 'Random variations in brightness, contrast, saturation, and hue'],
    ['GaussianBlur', 'Kernel: 5, Sigma: (0.1, 0.5), Probability: 0.2', 'Gaussian blur with kernel size 5 and 20% probability'],
    ['Normalize', 'Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]', 'ImageNet normalization applied to all images'],
]

# Column headers
headers = ['Augmentation Type', 'Parameters', 'Description']

# Create figure
fig, ax = plt.subplots(figsize=(16, 4.5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 4.5)
ax.axis('off')

# Title
ax.text(8, 4.1, 'Data Augmentation', ha='center', va='center',
        fontsize=20, weight='bold', color='#000000')

# Table dimensions
table_y_start = 3.6
row_height = 0.5
col_widths = [3.2, 5.5, 6.0]
col_positions = [0.5]
for width in col_widths[:-1]:
    col_positions.append(col_positions[-1] + width)

# Color scheme
header_color = '#424242'
header_text_color = 'white'
row_colors = ['#F5F5F5', '#FFFFFF']

# Draw header row
y = table_y_start
for i, (header, width) in enumerate(zip(headers, col_widths)):
    x = col_positions[i]

    # Header background
    rect = Rectangle((x, y - row_height), width, row_height,
                     facecolor=header_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)

    # Header text
    ax.text(x + width/2, y - row_height/2, header, ha='center', va='center',
           fontsize=11, weight='bold', color=header_text_color)

# Draw data rows
y = table_y_start - row_height

for idx, row_data in enumerate(augmentations):
    bg_color = row_colors[idx % 2]

    for col_idx, (cell_data, width) in enumerate(zip(row_data, col_widths)):
        x = col_positions[col_idx]

        # Cell background
        rect = Rectangle((x, y - row_height), width, row_height,
                         facecolor=bg_color, edgecolor='#BDBDBD', linewidth=1)
        ax.add_patch(rect)

        # Cell text
        fontsize = 9
        weight = 'bold' if col_idx == 0 else 'normal'

        # Use smaller line height for parameters column to prevent smudging
        if col_idx == 1:
            ax.text(x + width/2, y - row_height/2, cell_data, ha='center', va='center',
                   fontsize=fontsize, weight=weight, color='#000000', linespacing=0.9)
        else:
            ax.text(x + width/2, y - row_height/2, cell_data, ha='center', va='center',
                   fontsize=fontsize, weight=weight, color='#000000')

    y -= row_height

# Save figure
plt.tight_layout()
plt.savefig('Diagrams/augmentation_table.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Augmentation table saved to: Diagrams/augmentation_table.png")
plt.close()
