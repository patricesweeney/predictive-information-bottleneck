#!/usr/bin/env python3
"""
Demo script to test the pat_minimal matplotlib style.
Creates a scatter plot and CDF line plot to showcase the styling.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def main():
    """Create demo plots to verify the pat_minimal style."""
    
    # Generate sample data
    np.random.seed(42)
    n_points = 100
    
    # Data for scatter plot
    x = np.random.normal(0, 1, n_points)
    y = 2 * x + np.random.normal(0, 0.5, n_points)
    
    # Data for CDF
    data = np.random.gamma(2, 2, 1000)
    x_cdf = np.linspace(0, 15, 100)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.7))
    
    # Scatter plot
    ax1.scatter(x, y, alpha=0.7, s=20)
    ax1.set_xlabel(r'$X$ (input variable)')
    ax1.set_ylabel(r'$Y = 2X + \epsilon$')
    ax1.set_title('Scatter Plot with LaTeX Math')
    
    # Add a trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(x.min(), x.max(), 50)
    ax1.plot(x_trend, p(x_trend), '--', alpha=0.8, linewidth=1.5)
    
    # CDF plot
    # Empirical CDF
    sorted_data = np.sort(data)
    y_empirical = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax2.plot(sorted_data, y_empirical, label='Empirical CDF', linewidth=1.2)
    
    # Theoretical Gamma CDF
    theoretical_cdf = stats.gamma.cdf(x_cdf, 2, scale=2)
    ax2.plot(x_cdf, theoretical_cdf, label=r'Theoretical $\Gamma(2,2)$', linewidth=1.2)
    
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$F(x) = P(X \leq x)$')
    ax2.set_title('Cumulative Distribution Functions')
    ax2.legend()
    ax2.set_xlim(0, 12)
    
    # Add mathematical annotation
    ax2.text(0.6, 0.4, r'$F(x) = \int_{-\infty}^{x} f(t)\,dt$', 
             transform=ax2.transAxes, fontsize=9, 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs('examples/figs', exist_ok=True)
    
    # Save the plot
    plt.savefig('examples/figs/demo_style.pdf')
    plt.savefig('examples/figs/demo_style.png', dpi=200)
    
    print("Demo plots saved to:")
    print("  - examples/figs/demo_style.pdf")
    print("  - examples/figs/demo_style.png")
    
    # Show plot if running interactively
    plt.show()

if __name__ == '__main__':
    main()