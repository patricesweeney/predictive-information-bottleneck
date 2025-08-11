"""
Online Information Bottleneck Processing
=======================================

Consolidated online/streaming implementation using dependency injection.
Processes symbols one at a time with adaptive state growth.
"""

import numpy as np
from collections import defaultdict
from typing import Optional
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from interfaces import OnlineProcessor, AnalysisEngine, OptimizationEngine, AnalysisConfig


class OnlineProcessorImpl(OnlineProcessor):
    """
    Online information bottleneck processor using dependency injection.
    """
    
    def __init__(self,
                 analysis_engine: AnalysisEngine,
                 optimization_engine: OptimizationEngine,
                 config: AnalysisConfig = None,
                 inverse_temperature_beta: float = 60.0,
                 forgetting_factor: float = 1.0,
                 learning_rate: float = 0.05,
                 eigenvalue_check_interval: int = 500):
        """
        Initialize online processor with simplified dependencies.
        """
        super().__init__(analysis_engine, optimization_engine, config)
        
        self.alphabet_size = 2 ** self.config.future_window_length
        self.inverse_temperature_beta = inverse_temperature_beta
        self.forgetting_factor = forgetting_factor
        self.learning_rate = learning_rate
        self.eigenvalue_check_interval = eigenvalue_check_interval
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset online learning state."""
        np.random.seed(self.config.random_seed)
        
        # Empirical probability tracking
        self.symbol_counts = defaultdict(lambda: np.zeros(self.alphabet_size, float))
        self.context_totals = defaultdict(float)
        
        # Model parameters - start with single state
        self.emission_probabilities = np.full((1, self.alphabet_size), 1.0 / self.alphabet_size)
        self.posterior_distribution = None
        
        # Sliding window for past context
        self.past_context_buffer = []
        
        # Logging for analysis
        self.time_step = 0
        self.free_energy_log = []
        self.complexity_log = []
        self.accuracy_log = []
        self.state_count_log = []
        self.split_log = []
    
    def convert_window_to_index(self, window):
        """Convert binary window to integer index."""
        return int(''.join(map(str, window)), 2)
    
    def update_empirical_probabilities(self, past_context_index, future_symbol):
        """Update empirical probability estimates with new observation."""
        # Apply forgetting to existing counts
        self.symbol_counts[past_context_index] *= self.forgetting_factor
        self.context_totals[past_context_index] *= self.forgetting_factor
        
        # Add new observation
        self.symbol_counts[past_context_index][future_symbol] += 1.0
        self.context_totals[past_context_index] += 1.0
    
    def compute_current_empirical_distributions(self):
        """Compute current empirical P(future|past) and P(past) from accumulated counts."""
        # Get all observed past contexts
        observed_past_indices = sorted(self.context_totals.keys())
        
        if not observed_past_indices:
            return [], np.array([]), np.array([]).reshape(0, self.alphabet_size)
        
        # Marginal probabilities P(past)
        total_counts = sum(self.context_totals.values())
        past_probabilities = np.array([
            self.context_totals[idx] / total_counts for idx in observed_past_indices
        ])
        
        # Conditional probabilities P(future|past)
        future_conditional_probabilities = np.array([
            self.symbol_counts[idx] / max(self.context_totals[idx], 1e-10)
            for idx in observed_past_indices
        ])
        
        return observed_past_indices, past_probabilities, future_conditional_probabilities
    
    def process_symbol(self, new_symbol: int):
        """Process a single new symbol."""
        # Add to context buffer
        self.past_context_buffer.append(new_symbol)
        if len(self.past_context_buffer) > self.config.past_window_length + self.config.future_window_length:
            self.past_context_buffer.pop(0)
        
        # Check if we have enough context
        if len(self.past_context_buffer) < self.config.past_window_length + self.config.future_window_length:
            self.time_step += 1
            return
        
        # Extract past context and future window
        past_context = self.past_context_buffer[:self.config.past_window_length]
        future_window = self.past_context_buffer[self.config.past_window_length:]
        
        # Convert to indices
        past_context_index = self.convert_window_to_index(past_context)
        future_symbol = self.convert_window_to_index(future_window)
        
        # Update empirical probabilities
        self.update_empirical_probabilities(past_context_index, future_symbol)
        
        # Update model parameters with gradient step
        self._update_model_parameters()
        
        # Periodic state splitting check
        if self.time_step % self.eigenvalue_check_interval == 0:
            self._check_for_state_splits()
        
        # Log metrics
        self._log_current_metrics()
        
        self.time_step += 1
    
    def _update_model_parameters(self):
        """Update model parameters using gradient descent."""
        # Get current empirical distributions
        past_indices, past_probs, future_conditional = self.compute_current_empirical_distributions()
        
        if len(past_indices) == 0:
            return
        
        # Initialize posterior if needed
        if self.posterior_distribution is None:
            num_contexts = len(past_indices)
            num_states = self.emission_probabilities.shape[0]
            self.posterior_distribution = np.full((num_contexts, num_states), 1.0 / num_states)
        
        # Simple gradient update (simplified for brevity)
        # In practice, this would be a full EM step
        try:
            # Update emission probabilities based on posterior and data
            for state_idx in range(self.emission_probabilities.shape[0]):
                weighted_counts = np.zeros(self.alphabet_size)
                total_weight = 0.0
                
                for ctx_idx, past_idx in enumerate(past_indices):
                    weight = past_probs[ctx_idx] * self.posterior_distribution[ctx_idx, state_idx]
                    weighted_counts += weight * future_conditional[ctx_idx]
                    total_weight += weight
                
                if total_weight > 1e-10:
                    self.emission_probabilities[state_idx] = weighted_counts / total_weight
            
            # Normalize emission probabilities
            row_sums = self.emission_probabilities.sum(axis=1, keepdims=True)
            self.emission_probabilities = self.emission_probabilities / np.maximum(row_sums, 1e-10)
            
        except Exception:
            # Fallback to uniform if numerical issues
            self.emission_probabilities.fill(1.0 / self.alphabet_size)
    
    def _check_for_state_splits(self):
        """Check if any states should be split."""
        if self.emission_probabilities.shape[0] >= self.config.maximum_states_allowed:
            return
        
        # Get current empirical distributions
        past_indices, past_probs, future_conditional = self.compute_current_empirical_distributions()
        
        if len(past_indices) == 0:
            return
        
        try:
            # Attempt state split using injected splitter
            current_free_energy = self._compute_current_free_energy(past_probs, future_conditional)
            
            new_posterior, new_emission, new_free_energy, split_occurred = self.optimization_engine.attempt_state_split(
                past_probs,
                future_conditional,
                self.inverse_temperature_beta,
                self.posterior_distribution,
                self.emission_probabilities,
                current_free_energy
            )
            
            if split_occurred:
                self.posterior_distribution = new_posterior
                self.emission_probabilities = new_emission
                self.split_log.append(self.time_step)
                print(f"State split at step {self.time_step}: {self.emission_probabilities.shape[0]} states")
        
        except Exception:
            # Continue if splitting fails
            pass
    
    def _compute_current_free_energy(self, past_probs, future_conditional):
        """Compute current free energy."""
        try:
            free_energy, _, _, _, _ = self.analysis_engine.compute_variational_free_energy(
                past_probs,
                self.posterior_distribution,
                self.emission_probabilities,
                self.inverse_temperature_beta,
                future_conditional
            )
            return free_energy
        except Exception:
            return float('inf')
    
    def _log_current_metrics(self):
        """Log current metrics for analysis."""
        if self.time_step % 100 == 0:  # Log every 100 steps
            # Get current empirical distributions
            past_indices, past_probs, future_conditional = self.compute_current_empirical_distributions()
            
            if len(past_indices) > 0:
                try:
                    free_energy = self._compute_current_free_energy(past_probs, future_conditional)
                    self.free_energy_log.append(free_energy)
                    
                    # Compute complexity and accuracy
                    num_states = self.emission_probabilities.shape[0]
                    self.state_count_log.append(num_states)
                    
                    # Simple complexity estimate
                    complexity = np.log(num_states)
                    self.complexity_log.append(complexity)
                    
                    # Simple accuracy estimate (mutual information)
                    mutual_info = self.analysis_engine.compute_empirical_mutual_information(
                        past_probs, future_conditional
                    )
                    self.accuracy_log.append(mutual_info)
                    
                except Exception:
                    pass
    
    def get_analysis_results(self):
        """Get current analysis results."""
        return {
            'time_steps': list(range(0, len(self.free_energy_log) * 100, 100)),
            'free_energies': self.free_energy_log,
            'complexities': self.complexity_log,
            'accuracies': self.accuracy_log,
            'state_counts': self.state_count_log,
            'split_times': self.split_log,
            'current_num_states': self.emission_probabilities.shape[0],
            'total_contexts': len(self.context_totals)
        }
    
    def create_online_analysis_plot(self):
        """Create plots showing online learning progress using visualization module."""
        results = self.get_analysis_results()
        
        from visualization import create_online_analysis_plot
        return create_online_analysis_plot(results)


# Convenience function for quick online analysis
def run_online_analysis(process_generator, 
                       sequence_length: int = 10000,
                       process_name: str = "Unknown"):
    """
    Convenience function to run online analysis with standard implementations.
    """
    from interfaces import create_online_processor
    
    # Create processor with simplified dependencies
    processor = create_online_processor()
    
    # Generate sequence and process
    sequence = process_generator(sequence_length, seed=42)
    
    print(f"Processing {sequence_length} symbols from {process_name}...")
    for symbol in sequence:
        processor.process_symbol(symbol)
    
    results = processor.get_analysis_results()
    print(f"Final model: {results['current_num_states']} states, {results['total_contexts']} contexts")
    
    return processor, results


if __name__ == "__main__":
    # Run demo online analysis
    from processes import PROCESS_GENERATORS
    
    print("Running online information bottleneck analysis...")
    processor, results = run_online_analysis(
        PROCESS_GENERATORS["Golden-Mean"], 
        sequence_length=5000,
        process_name="Golden-Mean"
    )
    
    # Create plots
    processor.create_online_analysis_plot()
    
    from visualization import show_all_plots
    show_all_plots()
    
    print("Online analysis complete!")