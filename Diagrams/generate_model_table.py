"""Generate Model Architecture Table

Creates a simple table showing all 9 backbone architectures and their specifications.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Model data
models = [
    ['EfficientNet-B0', 'Traditional CNN', '4.01M', '8.02M', '4.52M', 'Compound-scaled CNN, balances depth/width/resolution'],
    ['ResNet-18', 'Traditional CNN', '11.18M', '22.36M', '11.69M', '18-layer residual network with skip connections'],
    ['ResNet-50', 'Traditional CNN', '23.51M', '47.02M', '24.54M', '50-layer residual network with bottleneck blocks'],
    ['ResNet-101', 'Traditional CNN', '42.50M', '85.00M', '43.53M', '101-layer residual network with bottleneck blocks'],
    ['ViT-B/16', 'Pure Transformer', '86.57M', '173.14M', '87.34M', 'Vision Transformer, 16×16 patches, global self-attention'],
    ['Swin-T', 'Pure Transformer', '27.52M', '55.04M', '28.29M', 'Hierarchical transformer, shifted-window attention, tiny variant'],
    ['Swin-B', 'Pure Transformer', '87.27M', '174.54M', '88.30M', 'Hierarchical transformer, shifted-window attention, base variant'],
    ['ConvNeXt-T', 'Modernized CNN', '27.82M', '55.64M', '28.59M', 'Modernized CNN with transformer design principles, tiny variant'],
    ['ConvNeXt-B', 'Modernized CNN', '88.02M', '176.04M', '89.05M', 'Modernized CNN with transformer design principles, base variant'],
]

# Column headers
headers = ['Model', 'Architecture Type', 'Parameters\n(TOP View)', 'Parameters\n(Late Fusion)', 'Parameters\n(Feature Fusion)', 'Key Characteristics']

# Create figure
fig, ax = plt.subplots(figsize=(18, 6))
ax.set_xlim(0, 18)
ax.set_ylim(0, 6)
ax.axis('off')

# Title
ax.text(9, 5.5, 'Model Architectures', ha='center', va='center',
        fontsize=20, weight='bold', color='#000000')

# Table dimensions
table_y_start = 4.8
row_height = 0.4
col_widths = [2.2, 2.2, 1.8, 1.8, 2.0, 5.5]
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
           fontsize=10, weight='bold', color=header_text_color)

# Draw data rows
y = table_y_start - row_height

for idx, row_data in enumerate(models):
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

        ax.text(x + width/2, y - row_height/2, cell_data, ha='center', va='center',
               fontsize=fontsize, weight=weight, color='#000000')

    y -= row_height

# Save figure
plt.tight_layout()
plt.savefig('Diagrams/model_table.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Model table saved to: Diagrams/model_table.png")
plt.close()
