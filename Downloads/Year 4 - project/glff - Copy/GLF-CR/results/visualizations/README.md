# Visualizations

Store output visualizations for qualitative analysis.

## Organization

```
visualizations/
├── baseline/           # Baseline model outputs
├── phase1/            # Phase 1 results
├── phase2/            # Phase 2 results
├── phase3/            # Phase 3 results
├── phase4/            # Phase 4 results
├── comparisons/       # Side-by-side comparisons
├── attention_maps/    # Attention visualizations
└── error_maps/        # Error analysis visualizations
```

## Naming Convention

- `{model_name}_{image_name}_input.png` - Input cloudy image
- `{model_name}_{image_name}_output.png` - Model output
- `{model_name}_{image_name}_target.png` - Ground truth
- `{model_name}_{image_name}_comparison.png` - Side-by-side
- `{model_name}_{image_name}_attention.png` - Attention maps
- `{model_name}_{image_name}_error.png` - Error map

## Quick View Script

```python
import matplotlib.pyplot as plt
from PIL import Image
import glob

# Load all comparisons
images = glob.glob('comparisons/*.png')
for img_path in images:
    img = Image.open(img_path)
    plt.figure(figsize=(15, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.title(img_path)
    plt.show()
```
