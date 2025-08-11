"""
Simplified Implementations for Consolidated Interfaces
=====================================================

Implementations for the consolidated interface design with reduced abstraction overhead.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from collections import defaultdict

try:
    from .interfaces import AnalysisEngine, OptimizationEngine
except ImportError:
    from interfaces import AnalysisEngine, OptimizationEngine


class StandardAnalysisEngine:
    """
    Standard implementation combining probability analysis and information calculations.
    
    Consolidates functionality from EmpiricalProbabilityAnalyzer + StandardInformationCalculator.
    """
    
    def extract_empirical_future_given_past(self, 
                                           sequence: np.ndarray,
                                           past_window_length: int,
                                           future_window_length: int) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """Extract P(future|past) from sequence data."""
        # Import the actual implementation
        from empirical_analysis import extract_empirical_future_given_past
        return extract_empirical_future_given_past(sequence, past_window_length, future_window_length)
    
    def compute_empirical_mutual_information(self,
                                           past_probabilities: np.ndarray,
                                           future_conditional_probabilities: np.ndarray) -> float:
        """Compute I(Past; Future)."""
        from empirical_analysis import compute_empirical_mutual_information
        return compute_empirical_mutual_information(past_probabilities, future_conditional_probabilities)
    
    def compute_kl_divergence(self, 
                            probability_p: np.ndarray, 
                            probability_q: np.ndarray) -> float:
        """Compute KL divergence D_KL(P||Q)."""
        from information_theory import compute_kl_divergence
        return compute_kl_divergence(probability_p, probability_q)
    
    def compute_variational_free_energy(self,
                                      past_probabilities: np.ndarray,
                                      posterior_distribution: np.ndarray,
                                      emission_probabilities: np.ndarray,
                                      inverse_temperature_beta: float,
                                      future_conditional_probabilities: np.ndarray) -> Tuple[float, float, float, float, float]:
        """Compute variational free energy with full decomposition."""
        from information_theory import compute_variational_free_energy
        return compute_variational_free_energy(
            past_probabilities, posterior_distribution, emission_probabilities,
            inverse_temperature_beta, future_conditional_probabilities
        )
    
    def compute_hessian_eigenvalue_for_state(self,
                                           target_state_index: int,
                                           past_probabilities: np.ndarray,
                                           posterior_distribution: np.ndarray,
                                           emission_probabilities: np.ndarray,
                                           future_conditional_probabilities: np.ndarray) -> float:
        """Compute largest eigenvalue for state stability analysis."""
        from information_theory import compute_hessian_eigenvalue_for_state
        return compute_hessian_eigenvalue_for_state(
            target_state_index, past_probabilities, posterior_distribution,
            emission_probabilities, future_conditional_probabilities
        )


class StandardOptimizationEngine:
    """
    Standard implementation combining EM optimization and state splitting.
    
    Consolidates functionality from StandardEMOptimizer + StandardStateSplitter.
    """
    
    def __init__(self, analysis_engine: AnalysisEngine):
        """Initialize with analysis engine dependency."""
        self.analysis_engine = analysis_engine
    
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
        from expectation_maximization import run_em_coordinate_ascent
        return run_em_coordinate_ascent(
            past_probabilities, future_conditional_probabilities, inverse_temperature_beta,
            initial_emission_probabilities, initial_posterior_distribution,
            maximum_iterations, convergence_tolerance, random_seed
        )
    
    def analyze_state_stability(self,
                               past_probabilities: np.ndarray,
                               posterior_distribution: np.ndarray,
                               emission_probabilities: np.ndarray,
                               future_conditional_probabilities: np.ndarray,
                               inverse_temperature_beta: float) -> Dict:
        """Analyze linear stability of states."""
        from state_splitting import analyze_state_stability
        return analyze_state_stability(
            past_probabilities, posterior_distribution, emission_probabilities,
            future_conditional_probabilities, inverse_temperature_beta
        )
    
    def attempt_state_split(self,
                          past_probabilities: np.ndarray,
                          future_conditional_probabilities: np.ndarray,
                          inverse_temperature_beta: float,
                          current_posterior_distribution: np.ndarray,
                          current_emission_probabilities: np.ndarray,
                          current_free_energy: float) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """Attempt to split most unstable state."""
        from state_splitting import attempt_state_split
        return attempt_state_split(
            past_probabilities, future_conditional_probabilities, inverse_temperature_beta,
            current_posterior_distribution, current_emission_probabilities, current_free_energy
        )