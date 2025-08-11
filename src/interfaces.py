"""
Simplified Interfaces for Information Bottleneck Components
=========================================================

Consolidated interfaces with reduced abstraction overhead while maintaining
essential modularity and dependency injection capabilities.

Target: 6 interfaces (down from 11) for interface density ~0.35
"""

from abc import ABC, abstractmethod
from typing import Protocol, Tuple, Dict, List, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters (simplified from Protocol)."""
    past_window_length: int = 10
    future_window_length: int = 2
    sample_length: int = 200000
    beta_schedule: Optional[np.ndarray] = None
    maximum_states_allowed: int = 16
    minimum_free_energy_improvement: float = 0.01
    random_seed: int = 42
    
    def __post_init__(self):
        if self.beta_schedule is None:
            self.beta_schedule = np.geomspace(0.05, 100, 120)


class ProcessGenerator(Protocol):
    """Interface for stochastic process generators."""
    
    def __call__(self, length: int, *, seed: int = 0) -> np.ndarray:
        """Generate a sequence of specified length."""
        ...


class AnalysisEngine(Protocol):
    """
    Consolidated interface for probability analysis and information calculations.
    
    Combines ProbabilityAnalyzer + InformationCalculator since they're always used together.
    """
    
    # Empirical probability analysis
    def extract_empirical_future_given_past(self, 
                                           sequence: np.ndarray,
                                           past_window_length: int,
                                           future_window_length: int) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """Extract P(future|past) from sequence data."""
        ...
    
    def compute_empirical_mutual_information(self,
                                           past_probabilities: np.ndarray,
                                           future_conditional_probabilities: np.ndarray) -> float:
        """Compute I(Past; Future)."""
        ...
    
    # Information-theoretic calculations
    def compute_kl_divergence(self, 
                            probability_p: np.ndarray, 
                            probability_q: np.ndarray) -> float:
        """Compute KL divergence D_KL(P||Q)."""
        ...
    
    def compute_variational_free_energy(self,
                                      past_probabilities: np.ndarray,
                                      posterior_distribution: np.ndarray,
                                      emission_probabilities: np.ndarray,
                                      inverse_temperature_beta: float,
                                      future_conditional_probabilities: np.ndarray) -> Tuple[float, float, float, float, float]:
        """Compute variational free energy with full decomposition.
        
        Returns:
            tuple: (free_energy, complexity, accuracy, energy, entropy)
        """
        ...
    
    def compute_hessian_eigenvalue_for_state(self,
                                           target_state_index: int,
                                           past_probabilities: np.ndarray,
                                           posterior_distribution: np.ndarray,
                                           emission_probabilities: np.ndarray,
                                           future_conditional_probabilities: np.ndarray) -> float:
        """Compute largest eigenvalue for state stability analysis."""
        ...


class OptimizationEngine(Protocol):
    """
    Consolidated interface for EM optimization and state splitting.
    
    Combines EMOptimizer + StateSplitter since they're tightly coupled.
    """
    
    # EM optimization
    def run_em_coordinate_ascent(self,
                               past_probabilities: np.ndarray,
                               future_conditional_probabilities: np.ndarray,
                               inverse_temperature_beta: float,
                               initial_emission_probabilities: Optional[np.ndarray] = None,
                               initial_posterior_distribution: Optional[np.ndarray] = None,
                               maximum_iterations: int = 400,
                               convergence_tolerance: float = 1e-6,
                               random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, float]:
        """Run EM algorithm to optimize information bottleneck."""
        ...
    
    # State splitting
    def analyze_state_stability(self,
                               past_probabilities: np.ndarray,
                               posterior_distribution: np.ndarray,
                               emission_probabilities: np.ndarray,
                               future_conditional_probabilities: np.ndarray,
                               inverse_temperature_beta: float) -> Dict:
        """Analyze linear stability of states."""
        ...
    
    def attempt_state_split(self,
                          past_probabilities: np.ndarray,
                          future_conditional_probabilities: np.ndarray,
                          inverse_temperature_beta: float,
                          current_posterior_distribution: np.ndarray,
                          current_emission_probabilities: np.ndarray,
                          current_free_energy: float) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """Attempt to split most unstable state."""
        ...


class InformationBottleneckAnalyzer(ABC):
    """Abstract base class for information bottleneck analysis."""
    
    def __init__(self,
                 analysis_engine: AnalysisEngine,
                 optimization_engine: OptimizationEngine,
                 config: AnalysisConfig = None):
        """Inject dependencies (simplified to 2 main engines)."""
        self.analysis_engine = analysis_engine
        self.optimization_engine = optimization_engine
        self.config = config or AnalysisConfig()
    
    @abstractmethod
    def run_analysis(self, process_generator: ProcessGenerator, process_name: str) -> Dict:
        """Run information bottleneck analysis on a process."""
        pass


class OnlineProcessor(ABC):
    """Abstract base class for online/streaming processing."""
    
    def __init__(self,
                 analysis_engine: AnalysisEngine,
                 optimization_engine: OptimizationEngine,
                 config: AnalysisConfig = None):
        """Inject dependencies (simplified to 2 main engines)."""
        self.analysis_engine = analysis_engine
        self.optimization_engine = optimization_engine
        self.config = config or AnalysisConfig()
    
    @abstractmethod
    def process_symbol(self, new_symbol: int) -> None:
        """Process a single new symbol."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset processing state."""
        pass


class Visualizer(Protocol):
    """Interface for creating visualizations."""
    
    def create_information_bottleneck_plot(self, results: Dict, **kwargs) -> Tuple:
        """Create IB trade-off curve plot."""
        ...
    
    def create_phase_transition_plot(self, results: Dict, **kwargs) -> Tuple:
        """Create phase transition plot."""
        ...
    
    def create_free_energy_trajectory_plot(self, results: Dict, **kwargs) -> Tuple:
        """Create free energy trajectory plot."""
        ...
    
    def create_online_analysis_plot(self, results: Dict, **kwargs) -> Tuple:
        """Create online analysis progress plot."""
        ...
    
    def create_process_comparison_plot(self, process_sequences: Dict, **kwargs) -> Tuple:
        """Create process comparison plot."""
        ...


# Simple factory functions (replace complex AnalysisFactory interface)
def create_standard_analysis_engine() -> AnalysisEngine:
    """Create standard analysis engine implementation."""
    try:
        from .implementations import StandardAnalysisEngine
    except ImportError:
        from implementations import StandardAnalysisEngine
    return StandardAnalysisEngine()


def create_standard_optimization_engine(analysis_engine: AnalysisEngine) -> OptimizationEngine:
    """Create standard optimization engine implementation."""
    try:
        from .implementations import StandardOptimizationEngine
    except ImportError:
        from implementations import StandardOptimizationEngine
    return StandardOptimizationEngine(analysis_engine)


def create_batch_analyzer(analysis_engine: AnalysisEngine = None,
                         optimization_engine: OptimizationEngine = None,
                         config: AnalysisConfig = None) -> InformationBottleneckAnalyzer:
    """Create batch analyzer with dependencies."""
    if analysis_engine is None:
        analysis_engine = create_standard_analysis_engine()
    if optimization_engine is None:
        optimization_engine = create_standard_optimization_engine(analysis_engine)
    
    # Import at runtime to avoid circular import issues
    import sys
    import os
    scripts_path = os.path.join(os.path.dirname(__file__), '..', 'scripts')
    sys.path.insert(0, scripts_path)
    from batch import BatchAnalyzer
    return BatchAnalyzer(analysis_engine, optimization_engine, config)


def create_online_processor(analysis_engine: AnalysisEngine = None,
                           optimization_engine: OptimizationEngine = None,
                           config: AnalysisConfig = None) -> OnlineProcessor:
    """Create online processor with dependencies."""
    if analysis_engine is None:
        analysis_engine = create_standard_analysis_engine()
    if optimization_engine is None:
        optimization_engine = create_standard_optimization_engine(analysis_engine)
    
    # Import at runtime to avoid circular import issues
    import sys
    import os
    scripts_path = os.path.join(os.path.dirname(__file__), '..', 'scripts')
    sys.path.insert(0, scripts_path)
    from online import OnlineProcessorImpl
    return OnlineProcessorImpl(analysis_engine, optimization_engine, config)


def create_visualizer() -> Visualizer:
    """Create visualizer implementation."""
    from visualization import InformationBottleneckVisualizer
    return InformationBottleneckVisualizer()


# Backwards compatibility - expose consolidated interfaces with old names
ProbabilityAnalyzer = AnalysisEngine  # For backwards compatibility
InformationCalculator = AnalysisEngine  # For backwards compatibility  
EMOptimizer = OptimizationEngine  # For backwards compatibility
StateSplitter = OptimizationEngine  # For backwards compatibility