# Matplotlib Style Guide

This project uses a custom matplotlib style (`pat_minimal.mplstyle`) that provides a clean, publication-ready appearance with Computer Modern fonts throughout.

## Quick Start

The style is automatically applied when you import the `src` package:

```python
import src  # Style is automatically applied
import matplotlib.pyplot as plt

# Your plots will now use the custom style
plt.plot([1, 2, 3], [1, 4, 9])
plt.show()
```

## Style Features

### Typography
- **Font Family**: Computer Modern Roman (LaTeX-style serif font)
- **Math Text**: Computer Modern math fonts (`mathtext.fontset: cm`)
- **Consistent Sizing**: Publication-appropriate font sizes (9-10.5pt)

### Visual Design
- **Clean Axes**: No top/right spines
- **Minimal Ticks**: Small outward ticks with minor ticks visible
- **Tight Layout**: Optimized for publication figures
- **Professional Colors**: Muted blue/red/grey color cycle

### Figure Settings
- **Default Size**: 3.4" × 2.7" (good for single-column papers)
- **High DPI**: 200 DPI for display, 300 DPI for saved figures
- **Tight Bounding**: Minimal whitespace around figures

## Usage Examples

### Basic Plotting
```python
import src
import matplotlib.pyplot as plt
import numpy as np

# Data
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# Plot with automatic styling
plt.figure()
plt.plot(x, y, label=r'$\sin(x)$')
plt.xlabel(r'$x$ (radians)')
plt.ylabel(r'$f(x)$')
plt.title('Trigonometric Function')
plt.legend()
plt.show()
```

### Information Theory Plots
```python
# Perfect for mathematical notation
plt.plot(beta_values, complexity, label=r'Complexity $I(X;Z)$')
plt.xlabel(r'Inverse Temperature $\beta$')
plt.ylabel(r'Information (bits)')
plt.title(r'Information Bottleneck: $\mathcal{L} = I(X;Z) - \beta I(Z;Y)$')
```

### Advanced Mathematical Expressions
```python
# Complex equations render beautifully
plt.title(r'Distribution: $p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$')
plt.text(0.5, 0.5, r'$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \, p(x) \, dx$', 
         transform=plt.gca().transAxes)
```

## Manual Style Application

If you need to manually apply the style:

```python
import matplotlib.pyplot as plt
from src.style_config import apply_project_style

apply_project_style()
```

## LaTeX Rendering (Optional)

For true LaTeX text rendering (requires TeX installation):

```python
from src.style_config import enable_full_latex

enable_full_latex()  # Enables LaTeX for ALL text, not just math
```

This will make every single character render through LaTeX using Computer Modern fonts.

## Saving Figures

Use the project's standard saving function:

```python
from src.style_config import save_figure

fig, ax = plt.subplots()
# ... create your plot ...
save_figure(fig, 'output/my_figure.pdf')  # Automatically uses optimal settings
```

## Color Palette

The default color cycle uses:
- **Primary Blue**: `#2F5C8A`
- **Accent Red**: `#C2504D` 
- **Dark Grey**: `#4F4F4F`
- **Light Grey**: `#8C8C8C`

Access programmatically:
```python
from src.style_config import get_project_colors
colors = get_project_colors()
```

## File Structure

```
styles/
  pat_minimal.mplstyle     # Main style file
matplotlibrc               # Project matplotlib config
src/
  style_config.py          # Style management utilities
examples/
  demo_style.py           # Basic style demonstration
  demo_computer_modern.py # Advanced typography demo
  figs/                   # Generated demo figures
```

## Demo Scripts

Run the demo scripts to see the style in action:

```bash
# Basic demo
python examples/demo_style.py

# Computer Modern typography showcase
python examples/demo_computer_modern.py
```

## Troubleshooting

### LaTeX Not Found
If you get "latex could not be found" when using `enable_full_latex()`:

1. **macOS**: Install MacTeX or BasicTeX
   ```bash
   brew install --cask mactex
   ```

2. **Linux**: Install TeXLive
   ```bash
   sudo apt-get install texlive-latex-base texlive-latex-extra
   ```

3. **Windows**: Install MiKTeX or TeXLive

### Font Issues
If Computer Modern fonts don't display correctly, the style will automatically fall back to other serif fonts. The math rendering will still use Computer Modern math fonts.

### Style Not Applied
If the style isn't applying automatically:

```python
import matplotlib.pyplot as plt
plt.style.use('styles/pat_minimal.mplstyle')
```

## Best Practices

1. **Mathematical Notation**: Always use raw strings for LaTeX math: `r'$\alpha$'`
2. **Figure Sizes**: Use the default size for consistency, or scale proportionally
3. **Colors**: Stick to the default color cycle for consistency across figures
4. **Saving**: Always save as both PDF (vector) and PNG (raster) formats
5. **LaTeX**: Only enable full LaTeX rendering for final publication figures

## Integration with Existing Code

The style integrates seamlessly with the project's visualization module:

```python
from src.visualization import create_information_bottleneck_plot

# All visualization functions automatically use the custom style
fig, ax = create_information_bottleneck_plot(results)
```

This ensures all plots throughout the project maintain consistent, publication-ready appearance with Computer Modern typography.