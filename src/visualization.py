"""
Visualization and Plotting for Information Bottleneck Analysis
============================================================

Consolidated plotting and visualization functionality for both batch and online
information bottleneck analysis. Provides clean separation between analysis 
logic and visualization concerns.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any


class InformationBottleneckVisualizer:
    """
    Visualizer for information bottleneck analysis results.
    
    Handles both batch and online analysis visualization with consistent
    styling and layout.
    """
    
    def __init__(self, 
                 figsize_default: Tuple[int, int] = (10, 6),
                 style: str = 'default',
                 dpi: int = 100):
        """
        Initialize visualizer with styling options.
        
        Args:
            figsize_default: Default figure size for plots
            style: Matplotlib style to use
            dpi: Figure resolution
        """
        self.figsize_default = figsize_default
        self.style = style
        self.dpi = dpi
        
        # Set matplotlib style
        if style != 'default':
            plt.style.use(style)
    
    def create_information_bottleneck_plot(self, 
                                         results: Dict[str, Dict],
                                         figsize: Optional[Tuple[int, int]] = None,
                                         title: str = "Information Bottleneck Trade-off Curves") -> Tuple[plt.Figure, plt.Axes]:
        """
        Create information bottleneck trade-off plot showing complexity vs accuracy.
        
        Args:
            results: Dictionary mapping process names to analysis results
            figsize: Figure size (width, height)
            title: Plot title
            
        Returns:
            Tuple of (figure, axes)
        """
        if figsize is None:
            figsize = self.figsize_default
        
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        for process_name, result in results.items():
            complexities = result['complexities']
            accuracies = result['accuracies']
            ax.plot(complexities, accuracies, 'o-', label=process_name, alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Complexity I(X;Z)', fontsize=12)
        ax.set_ylabel('Accuracy I(Z;Y)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add some styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig, ax
    
    def create_vfe_decomposition_plot(self,
                                     results: Dict[str, Dict],
                                     figsize: Optional[Tuple[int, int]] = None,
                                     title: str = "Variational Free Energy Decompositions") -> Tuple[plt.Figure, Tuple[Tuple[plt.Axes, plt.Axes], Tuple[plt.Axes, plt.Axes]]]:
        """
        Create comprehensive VFE decomposition plot showing both:
        1. Accuracy/Complexity decomposition: F = Complexity - β × Accuracy
        2. Energy/Entropy decomposition: F = Energy - T × Entropy
        
        Args:
            results: Dictionary mapping process names to analysis results
            figsize: Figure size (width, height)
            title: Plot title
            
        Returns:
            Tuple of (figure, ((acc_ax, comp_ax), (energy_ax, entropy_ax)))
        """
        if figsize is None:
            figsize = (self.figsize_default[0] * 1.5, self.figsize_default[1] * 1.8)
        
        fig = plt.figure(figsize=figsize, dpi=self.dpi)
        
        # Create 2x2 subplot grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Accuracy/Complexity decomposition (top row)
        acc_ax = fig.add_subplot(gs[0, 0])
        comp_ax = fig.add_subplot(gs[0, 1])
        
        # Energy/Entropy decomposition (bottom row)  
        energy_ax = fig.add_subplot(gs[1, 0])
        entropy_ax = fig.add_subplot(gs[1, 1])
        
        for process_name, result in results.items():
            beta_values = result['beta_values']
            accuracies = result['accuracies']
            complexities = result['complexities']
            energies = result['energies']
            entropies = result['entropies']
            
            # Plot accuracy/complexity decomposition
            acc_ax.semilogx(beta_values, accuracies, 'o-', label=process_name, alpha=0.8, linewidth=2)
            comp_ax.semilogx(beta_values, complexities, 'o-', label=process_name, alpha=0.8, linewidth=2)
            
            # Plot energy/entropy decomposition
            energy_ax.semilogx(beta_values, energies, 'o-', label=process_name, alpha=0.8, linewidth=2)
            entropy_ax.semilogx(beta_values, entropies, 'o-', label=process_name, alpha=0.8, linewidth=2)
        
        # Style accuracy plot
        acc_ax.set_title('Accuracy: E[log P(future|state)]', fontsize=12, fontweight='bold')
        acc_ax.set_ylabel('Accuracy', fontsize=11)
        acc_ax.legend(fontsize=9)
        acc_ax.grid(True, alpha=0.3)
        acc_ax.spines['top'].set_visible(False)
        acc_ax.spines['right'].set_visible(False)
        
        # Style complexity plot
        comp_ax.set_title('Complexity: I(Past; State)', fontsize=12, fontweight='bold')
        comp_ax.set_ylabel('Complexity', fontsize=11)
        comp_ax.legend(fontsize=9)
        comp_ax.grid(True, alpha=0.3)
        comp_ax.spines['top'].set_visible(False)
        comp_ax.spines['right'].set_visible(False)
        
        # Style energy plot
        energy_ax.set_title('Energy: -E[log P(past, future)]', fontsize=12, fontweight='bold')
        energy_ax.set_xlabel('Inverse Temperature β', fontsize=11)
        energy_ax.set_ylabel('Energy', fontsize=11)
        energy_ax.legend(fontsize=9)
        energy_ax.grid(True, alpha=0.3)
        energy_ax.spines['top'].set_visible(False)
        energy_ax.spines['right'].set_visible(False)
        
        # Style entropy plot
        entropy_ax.set_title('Entropy: H[q(state|past)]', fontsize=12, fontweight='bold')
        entropy_ax.set_xlabel('Inverse Temperature β', fontsize=11)
        entropy_ax.set_ylabel('Entropy', fontsize=11)
        entropy_ax.legend(fontsize=9)
        entropy_ax.grid(True, alpha=0.3)
        entropy_ax.spines['top'].set_visible(False)
        entropy_ax.spines['right'].set_visible(False)
        
        # Add main title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        
        return fig, ((acc_ax, comp_ax), (energy_ax, entropy_ax))

    def create_phase_transition_plot(self, 
                                   results: Dict[str, Dict],
                                   figsize: Optional[Tuple[int, int]] = None,
                                   title: str = "Phase Transitions vs Inverse Temperature") -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """
        Create phase transition plot showing state count and free energy vs beta.
        
        Args:
            results: Dictionary mapping process names to analysis results
            figsize: Figure size (width, height)
            title: Plot title
            
        Returns:
            Tuple of (figure, (ax1, ax2))
        """
        if figsize is None:
            figsize = (self.figsize_default[0], self.figsize_default[1] * 1.3)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, dpi=self.dpi)
        
        for process_name, result in results.items():
            beta_values = result['beta_values']
            num_states = result['num_states']
            free_energies = result['free_energies']
            
            ax1.semilogx(beta_values, num_states, 'o-', label=process_name, alpha=0.8, linewidth=2)
            ax2.semilogx(beta_values, free_energies, 'o-', label=process_name, alpha=0.8, linewidth=2)
        
        ax1.set_ylabel('Number of States', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        ax2.set_xlabel('Inverse Temperature β', fontsize=12)
        ax2.set_ylabel('Free Energy', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def create_free_energy_trajectory_plot(self, 
                                         results: Dict[str, Dict],
                                         figsize: Optional[Tuple[int, int]] = None,
                                         title: str = "Free Energy Trajectories") -> Tuple[plt.Figure, plt.Axes]:
        """
        Create plot showing free energy trajectories across beta values.
        
        Args:
            results: Dictionary mapping process names to analysis results
            figsize: Figure size (width, height)
            title: Plot title
            
        Returns:
            Tuple of (figure, axes)
        """
        if figsize is None:
            figsize = self.figsize_default
        
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        for process_name, result in results.items():
            beta_values = result['beta_values']
            free_energies = result['free_energies']
            ax.semilogx(beta_values, free_energies, 'o-', label=process_name, alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Inverse Temperature β', fontsize=12)
        ax.set_ylabel('Free Energy', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig, ax
    
    def create_online_analysis_plot(self, 
                                  results: Dict[str, Any],
                                  figsize: Optional[Tuple[int, int]] = None,
                                  title: str = "Online Learning Progress") -> Tuple[plt.Figure, Tuple[Tuple[plt.Axes, plt.Axes], Tuple[plt.Axes, plt.Axes]]]:
        """
        Create comprehensive plot showing online learning progress.
        
        Args:
            results: Online analysis results dictionary
            figsize: Figure size (width, height)
            title: Overall plot title
            
        Returns:
            Tuple of (figure, ((ax1, ax2), (ax3, ax4)))
        """
        if figsize is None:
            figsize = (12, 8)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize, dpi=self.dpi)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        time_steps = results['time_steps']
        
        # Free energy over time
        ax1.plot(time_steps, results['free_energies'], 'b-', linewidth=2)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Free Energy')
        ax1.set_title('Free Energy Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Number of states over time
        ax2.plot(time_steps, results['state_counts'], 'g-', linewidth=2)
        for split_time in results['split_times']:
            ax2.axvline(split_time, color='red', alpha=0.7, linestyle='--', linewidth=1)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Number of States')
        ax2.set_title('State Growth (red lines = splits)')
        ax2.grid(True, alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Complexity over time
        ax3.plot(time_steps, results['complexities'], 'orange', linewidth=2)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Complexity')
        ax3.set_title('Model Complexity')
        ax3.grid(True, alpha=0.3)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # Accuracy over time
        ax4.plot(time_steps, results['accuracies'], 'purple', linewidth=2)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Prediction Accuracy')
        ax4.grid(True, alpha=0.3)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig, ((ax1, ax2), (ax3, ax4))
    
    def create_process_comparison_plot(self, 
                                     process_sequences: Dict[str, np.ndarray],
                                     max_length: int = 100,
                                     figsize: Optional[Tuple[int, int]] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create visual comparison of different process sequences.
        
        Args:
            process_sequences: Dictionary mapping process names to sequences
            max_length: Maximum sequence length to display
            figsize: Figure size (width, height)
            
        Returns:
            Tuple of (figure, axes)
        """
        if figsize is None:
            figsize = (14, len(process_sequences) * 1.5)
        
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        y_positions = []
        process_names = list(process_sequences.keys())
        
        for i, (process_name, sequence) in enumerate(process_sequences.items()):
            y_pos = len(process_names) - i - 1
            y_positions.append(y_pos)
            
            # Truncate sequence if too long
            display_sequence = sequence[:max_length]
            
            # Create visualization using scatter plot for binary sequences
            ones_positions = np.where(display_sequence == 1)[0]
            zeros_positions = np.where(display_sequence == 0)[0]
            
            ax.scatter(ones_positions, [y_pos] * len(ones_positions), 
                      c='red', s=20, marker='s', alpha=0.8, label='1' if i == 0 else "")
            ax.scatter(zeros_positions, [y_pos] * len(zeros_positions), 
                      c='blue', s=20, marker='s', alpha=0.8, label='0' if i == 0 else "")
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(process_names)
        ax.set_xlabel(f'Time Step (showing first {max_length} symbols)')
        ax.set_title('Process Sequence Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend()
        
        plt.tight_layout()
        return fig, ax
    
    def create_comprehensive_analysis_dashboard(self, 
                                              batch_results: Dict[str, Dict],
                                              online_results: Optional[Dict[str, Any]] = None,
                                              figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        Create comprehensive dashboard with multiple analysis plots.
        
        Args:
            batch_results: Batch analysis results
            online_results: Optional online analysis results
            figsize: Figure size (width, height)
            
        Returns:
            Figure object
        """
        if online_results is not None:
            fig = plt.figure(figsize=figsize, dpi=self.dpi)
            
            # Create subplot layout: 2x3 grid
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            
            # Information bottleneck plot
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_ib_curve_on_axis(batch_results, ax1)
            
            # Phase transition plot (states)
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_phase_transition_states_on_axis(batch_results, ax2)
            
            # Online free energy
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_online_free_energy_on_axis(online_results, ax3)
            
            # Online state growth
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_online_states_on_axis(online_results, ax4)
            
            # Combined complexity comparison
            ax5 = fig.add_subplot(gs[2, :])
            self._plot_complexity_comparison_on_axis(batch_results, ax5)
            
        else:
            # Batch-only dashboard
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize, dpi=self.dpi)
            
            self._plot_ib_curve_on_axis(batch_results, ax1)
            self._plot_phase_transition_states_on_axis(batch_results, ax2)
            self._plot_phase_transition_energy_on_axis(batch_results, ax3)
            self._plot_complexity_comparison_on_axis(batch_results, ax4)
        
        fig.suptitle('Information Bottleneck Analysis Dashboard', fontsize=18, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _plot_ib_curve_on_axis(self, results: Dict[str, Dict], ax: plt.Axes):
        """Helper method to plot IB curve on given axis."""
        for process_name, result in results.items():
            complexities = result['complexities']
            accuracies = result['accuracies']
            ax.plot(complexities, accuracies, 'o-', label=process_name, alpha=0.8)
        
        ax.set_xlabel('Complexity I(X;Z)')
        ax.set_ylabel('Accuracy I(Z;Y)')
        ax.set_title('Information Bottleneck Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_phase_transition_states_on_axis(self, results: Dict[str, Dict], ax: plt.Axes):
        """Helper method to plot phase transition states on given axis."""
        for process_name, result in results.items():
            beta_values = result['beta_values']
            num_states = result['num_states']
            ax.semilogx(beta_values, num_states, 'o-', label=process_name, alpha=0.8)
        
        ax.set_xlabel('Inverse Temperature β')
        ax.set_ylabel('Number of States')
        ax.set_title('Phase Transitions')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_phase_transition_energy_on_axis(self, results: Dict[str, Dict], ax: plt.Axes):
        """Helper method to plot phase transition energy on given axis."""
        for process_name, result in results.items():
            beta_values = result['beta_values']
            free_energies = result['free_energies']
            ax.semilogx(beta_values, free_energies, 'o-', label=process_name, alpha=0.8)
        
        ax.set_xlabel('Inverse Temperature β')
        ax.set_ylabel('Free Energy')
        ax.set_title('Free Energy Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_online_free_energy_on_axis(self, results: Dict[str, Any], ax: plt.Axes):
        """Helper method to plot online free energy on given axis."""
        time_steps = results['time_steps']
        ax.plot(time_steps, results['free_energies'], 'b-', linewidth=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Free Energy')
        ax.set_title('Online Free Energy')
        ax.grid(True, alpha=0.3)
    
    def _plot_online_states_on_axis(self, results: Dict[str, Any], ax: plt.Axes):
        """Helper method to plot online state growth on given axis."""
        time_steps = results['time_steps']
        ax.plot(time_steps, results['state_counts'], 'g-', linewidth=2)
        for split_time in results['split_times']:
            ax.axvline(split_time, color='red', alpha=0.7, linestyle='--')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Number of States')
        ax.set_title('Online State Growth')
        ax.grid(True, alpha=0.3)
    
    def _plot_complexity_comparison_on_axis(self, results: Dict[str, Dict], ax: plt.Axes):
        """Helper method to plot complexity comparison on given axis."""
        process_names = list(results.keys())
        final_complexities = [results[name]['complexities'][-1] for name in process_names]
        
        bars = ax.bar(process_names, final_complexities, alpha=0.7)
        ax.set_ylabel('Final Complexity (bits)')
        ax.set_title('Process Complexity Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if needed
        if len(process_names) > 3:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


# Global visualizer instance for convenience
default_visualizer = InformationBottleneckVisualizer()

# Convenience functions that use the default visualizer
def create_information_bottleneck_plot(results: Dict[str, Dict], **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Convenience function for creating IB plot with default visualizer."""
    return default_visualizer.create_information_bottleneck_plot(results, **kwargs)

def create_phase_transition_plot(results: Dict[str, Dict], **kwargs) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Convenience function for creating phase transition plot with default visualizer."""
    return default_visualizer.create_phase_transition_plot(results, **kwargs)

def create_online_analysis_plot(results: Dict[str, Any], **kwargs) -> Tuple[plt.Figure, Tuple[Tuple[plt.Axes, plt.Axes], Tuple[plt.Axes, plt.Axes]]]:
    """Convenience function for creating online analysis plot with default visualizer."""
    return default_visualizer.create_online_analysis_plot(results, **kwargs)

def create_free_energy_plot(results: Dict[str, Dict], **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Convenience function for creating free energy plot with default visualizer."""
    return default_visualizer.create_free_energy_trajectory_plot(results, **kwargs)

def create_process_comparison_plot(process_sequences: Dict[str, np.ndarray], **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Convenience function for creating process comparison plot with default visualizer."""
    return default_visualizer.create_process_comparison_plot(process_sequences, **kwargs)

def create_vfe_decomposition_plot(results: Dict[str, Dict], **kwargs) -> Tuple[plt.Figure, Tuple[Tuple[plt.Axes, plt.Axes], Tuple[plt.Axes, plt.Axes]]]:
    """Convenience function for creating VFE decomposition plot with default visualizer."""
    return default_visualizer.create_vfe_decomposition_plot(results, **kwargs)

def show_all_plots(save_to_dir: str = None):
    """Convenience function to display all created plots or save them."""
    if save_to_dir:
        # Save all figures instead of showing
        import os
        os.makedirs(save_to_dir, exist_ok=True)
        
        for i, fig_num in enumerate(plt.get_fignums()):
            fig = plt.figure(fig_num)
            filename = os.path.join(save_to_dir, f'figure_{i+1}.png')
            fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved figure to {filename}")
        print(f"All figures saved to {save_to_dir}/")
    else:
        plt.show()


if __name__ == "__main__":
    # Demo of visualization capabilities
    print("Information Bottleneck Visualization Module")
    print("Use the convenience functions or InformationBottleneckVisualizer class")
    print("for creating publication-ready plots of analysis results.")