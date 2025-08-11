#!/usr/bin/env python3
"""
Computer Modern Font Demo
========================

Demonstrates the Computer Modern font styling with both matplotlib's 
Computer Modern math fonts and optional true LaTeX rendering.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys

# Add src to path to import style configuration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from style_config import apply_project_style, enable_full_latex

def create_computer_modern_demo(use_latex=False):
    """
    Create demo plots showcasing Computer Modern fonts.
    
    Args:
        use_latex: If True, enable full LaTeX text rendering
    """
    
    # Apply styling
    apply_project_style()
    
    if use_latex:
        print("Enabling full LaTeX rendering...")
        enable_full_latex()
    
    # Generate sample data
    np.random.seed(42)
    
    # Create figure with multiple subplots to show different text elements
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # Subplot 1: Mathematical expressions
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.linspace(0, 2*np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    ax1.plot(x, y1, label=r'$\sin(x)$', linewidth=1.5)
    ax1.plot(x, y2, label=r'$\cos(x)$', linewidth=1.5)
    ax1.set_xlabel(r'$x$ (radians)')
    ax1.set_ylabel(r'$f(x)$')
    ax1.set_title(r'Trigonometric Functions: $\sin(x)$ and $\cos(x)$')
    ax1.legend()

    
    # Add mathematical annotation
    ax1.annotate(r'$\sin^2(x) + \cos^2(x) = 1$', 
                xy=(np.pi/4, np.sin(np.pi/4)), xytext=(np.pi, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, ha='center')
    
    # Subplot 2: Information theory notation
    ax2 = fig.add_subplot(gs[0, 1])
    beta_values = np.logspace(-2, 2, 50)
    complexity = 2 * np.log(1 + np.exp(-beta_values))
    accuracy = beta_values / (1 + beta_values)
    
    ax2.semilogx(beta_values, complexity, label=r'Complexity $I(X;Z)$')
    ax2.semilogx(beta_values, accuracy, label=r'Accuracy $I(Z;Y)$')
    ax2.set_xlabel(r'Inverse Temperature $\beta$')
    ax2.set_ylabel(r'Information (bits)')
    ax2.set_title(r'Information Bottleneck: $\mathcal{L} = I(X;Z) - \beta I(Z;Y)$')
    ax2.legend()

    
    # Subplot 3: Greek letters and special symbols
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Generate some statistical data
    x_data = np.random.normal(0, 1, 1000)
    
    # Plot histogram
    n, bins, patches = ax3.hist(x_data, bins=30, density=True, alpha=0.7, 
                               color='#2F5C8A', edgecolor='black', linewidth=0.5)
    
    # Overlay theoretical normal distribution
    x_theory = np.linspace(-4, 4, 100)
    y_theory = stats.norm.pdf(x_theory, 0, 1)
    ax3.plot(x_theory, y_theory, 'r-', linewidth=2, 
             label=r'$\mathcal{N}(\mu=0, \sigma^2=1)$')
    
    ax3.set_xlabel(r'$x$')
    ax3.set_ylabel(r'Probability Density $p(x)$')
    ax3.set_title(r'Normal Distribution: $p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$')
    ax3.legend()

    
    # Subplot 4: Complex mathematical expressions
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Plot some interesting mathematical function
    t = np.linspace(0, 4*np.pi, 1000)
    spiral_r = np.exp(-t/10)
    x_spiral = spiral_r * np.cos(t)
    y_spiral = spiral_r * np.sin(t)
    
    ax4.plot(x_spiral, y_spiral, linewidth=1.5, color='#C2504D')
    ax4.set_xlabel(r'$x = r(\theta) \cos(\theta)$')
    ax4.set_ylabel(r'$y = r(\theta) \sin(\theta)$')
    ax4.set_title(r'Logarithmic Spiral: $r(\theta) = e^{-\alpha\theta}$')
    ax4.set_aspect('equal')

    
    # Add equation as text
    ax4.text(0.05, 0.95, r'$\alpha = \frac{1}{10}$', transform=ax4.transAxes,
             verticalalignment='top', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add overall title with complex mathematical expression
    fig.suptitle(r'Computer Modern Typography Demo: $\mathbb{E}[X] = \int_{-\infty}^{\infty} x \, p(x) \, dx$', 
                 fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    # Save the plots
    os.makedirs('examples/figs', exist_ok=True)
    
    suffix = '_latex' if use_latex else '_matplotlib'
    pdf_name = f'examples/figs/computer_modern_demo{suffix}.pdf'
    png_name = f'examples/figs/computer_modern_demo{suffix}.png'
    
    plt.savefig(pdf_name)
    plt.savefig(png_name, dpi=200)
    
    print(f"Computer Modern demo saved to:")
    print(f"  - {pdf_name}")
    print(f"  - {png_name}")
    
    if use_latex:
        print("  (Using full LaTeX rendering)")
    else:
        print("  (Using matplotlib's Computer Modern math fonts)")
    
    plt.show()

def main():
    """Run both matplotlib and LaTeX versions for comparison."""
    
    print("=== Computer Modern Typography Demo ===\n")
    
    # First, create version with matplotlib's Computer Modern
    print("1. Creating demo with matplotlib's Computer Modern fonts...")
    create_computer_modern_demo(use_latex=False)
    
    # Ask user if they want to try LaTeX version
    try_latex = input("\nDo you want to try the full LaTeX version? (y/N): ").lower().startswith('y')
    
    if try_latex:
        print("\n2. Creating demo with full LaTeX rendering...")
        try:
            create_computer_modern_demo(use_latex=True)
        except Exception as e:
            print(f"LaTeX rendering failed: {e}")
            print("Make sure you have a TeX distribution installed (e.g., TeXLive, MiKTeX)")
    
    print("\nDemo complete!")

if __name__ == '__main__':
    main()