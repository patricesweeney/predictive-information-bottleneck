"""
Matplotlib Style Configuration for the Project
=============================================

Centralized style configuration that ensures all plots in the project
use the pat_minimal style for publication-ready figures.
"""

import matplotlib.pyplot as plt
import os
from pathlib import Path

def apply_project_style():
    """
    Apply the pat_minimal matplotlib style to all plots.
    
    This function should be called at the beginning of any script
    that creates plots to ensure consistent styling.
    """
    try:
        # Get the project root directory
        project_root = Path(__file__).parent.parent
        style_path = project_root / 'styles' / 'pat_minimal.mplstyle'
        
        if style_path.exists():
            plt.style.use(str(style_path))
            print(f"Applied custom style: {style_path}")
        else:
            print(f"Warning: Custom style file not found at {style_path}")
            print("Using default matplotlib style")
            
    except Exception as e:
        print(f"Warning: Could not apply custom style: {e}")
        print("Using default matplotlib style")

def configure_latex_math():
    """
    Configure matplotlib for LaTeX-like math rendering.
    
    Call this function if you want to enable true LaTeX text rendering
    (requires a TeX distribution to be installed).
    """
    try:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb,amsfonts}'
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman']
        print("Enabled LaTeX text rendering with Computer Modern")
    except Exception as e:
        print(f"Warning: Could not enable LaTeX rendering: {e}")
        print("Using Computer Modern math fonts instead")

def enable_full_latex():
    """
    Enable full LaTeX text rendering for publication-quality output.
    
    This will make ALL text (including labels, titles, tick labels) 
    render through LaTeX using Computer Modern fonts.
    """
    configure_latex_math()

def get_project_colors():
    """
    Get the standard color palette used in the project.
    
    Returns:
        list: List of hex color codes for consistent plotting
    """
    return ['#2F5C8A', '#C2504D', '#4F4F4F', '#8C8C8C']

def save_figure(fig, filename, dpi=300, bbox_inches='tight', pad_inches=0.02):
    """
    Save figure with project-standard settings.
    
    Args:
        fig: matplotlib Figure object
        filename: Output filename (with or without extension)
        dpi: Resolution for saved figure
        bbox_inches: Bounding box setting
        pad_inches: Padding around the figure
    """
    # Ensure the filename has an extension
    if '.' not in filename:
        filename += '.pdf'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    print(f"Saved figure: {filename}")

# Automatically apply style when this module is imported
apply_project_style()