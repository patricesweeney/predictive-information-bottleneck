"""
Predictive Information Bottleneck Analysis
==========================================

A clean, minimal implementation of information bottleneck methods for
analyzing stochastic processes with dependency injection and composition.

Main Components:
- processes: Stochastic process generators
- batch: Batch information bottleneck analysis  
- online: Online/streaming information bottleneck processing
- interfaces: Abstract interfaces for dependency injection
- implementations: Concrete implementations of all interfaces
- demo: Comprehensive demonstrations

Quick Start:
    from src.batch import run_default_analysis
    from src.processes import PROCESS_GENERATORS
    
    analyzer, results = run_default_analysis(
        selected_processes=["Golden-Mean", "IID(p=0.5)"]
    )
"""

# Apply project-wide matplotlib styling
try:
    from .style_config import apply_project_style
    apply_project_style()
except ImportError:
    pass  # Style config not available, continue with default styling

# Core building blocks
from .interfaces import (
    ProcessGenerator,
    AnalysisEngine,
    OptimizationEngine, 
    InformationBottleneckAnalyzer,
    OnlineProcessor,
    Visualizer,
    AnalysisConfig,
    create_batch_analyzer,
    create_online_processor,
    create_visualizer
)

from .implementations import (
    StandardAnalysisEngine,
    StandardOptimizationEngine
)

# Process generators
from .processes import PROCESS_GENERATORS

# High-level analysis classes (scripts moved to scripts/ directory)
# For convenience, import the classes from the scripts
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from batch import BatchAnalyzer, run_default_analysis
from online import OnlineProcessorImpl, run_online_analysis

# Visualization
from .visualization import (
    InformationBottleneckVisualizer,
    create_information_bottleneck_plot,
    create_phase_transition_plot,
    create_online_analysis_plot,
    create_free_energy_plot,
    show_all_plots
)

# Core algorithms (if needed directly)
from .empirical_analysis import (
    extract_empirical_future_given_past,
    compute_empirical_mutual_information
)
from .information_theory import (
    compute_kl_divergence,
    compute_variational_free_energy
)
from .expectation_maximization import run_em_coordinate_ascent
from .state_splitting import run_adaptive_state_growth

__version__ = "2.0.0"
__author__ = "Predictive Information Bottleneck Project"

# Main public API
__all__ = [
    # High-level functions (most users start here)
    "run_default_analysis",
    "run_online_analysis", 
    "PROCESS_GENERATORS",
    
    # Main classes
    "BatchAnalyzer",
    "OnlineProcessorImpl",
    "StandardAnalysisFactory",
    "InformationBottleneckVisualizer",
    
    # Interfaces (for advanced users doing dependency injection)
    "ProcessGenerator",
    "AnalysisEngine",
    "OptimizationEngine",
    "InformationBottleneckAnalyzer", 
    "OnlineProcessor",
    "Visualizer",
    "AnalysisConfig",
    "create_batch_analyzer",
    "create_online_processor", 
    "create_visualizer",
    
    # Core algorithms (for researchers extending the library)
    "extract_empirical_future_given_past",
    "compute_empirical_mutual_information",
    "compute_kl_divergence",
    "compute_variational_free_energy",
    "run_em_coordinate_ascent",
    "run_adaptive_state_growth",
    
    # Visualization functions
    "create_information_bottleneck_plot",
    "create_phase_transition_plot", 
    "create_online_analysis_plot",
    "create_free_energy_plot",
    "show_all_plots",
]


def quick_start_example():
    """
    Run a quick demonstration of the library capabilities.
    """
    print("Running quick start example...")
    
    # Batch analysis
    analyzer, results = run_default_analysis(
        selected_processes=["Golden-Mean", "IID(p=0.5)"]
    )
    
    # Show results
    for process_name, result in results.items():
        print(f"{process_name}: {result['num_states'][-1]} states, "
              f"MI={result['empirical_mutual_information']:.3f}")
    
    return analyzer, results


if __name__ == "__main__":
    # Quick demonstration if run directly
    quick_start_example()