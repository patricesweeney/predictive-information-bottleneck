"""
Batch Information Bottleneck Analysis
====================================

Consolidated batch analysis implementation using dependency injection.
Analyzes stochastic processes to find optimal information bottleneck representations.
"""

import numpy as np
from typing import Dict, List, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from interfaces import (
    InformationBottleneckAnalyzer, AnalysisEngine, 
    OptimizationEngine, ProcessGenerator, AnalysisConfig
)


class BatchAnalyzer(InformationBottleneckAnalyzer):
    """
    Batch information bottleneck analyzer using dependency injection.
    """
    
    def __init__(self,
                 analysis_engine: AnalysisEngine,
                 optimization_engine: OptimizationEngine,
                 config: AnalysisConfig = None):
        """
        Initialize with simplified dependencies.
        """
        super().__init__(analysis_engine, optimization_engine, config)
        
        # Config is now handled by parent class
        self.beta_schedule = self.config.beta_schedule
        
        # Storage for results
        self.results = {}
        self.analysis_complete = False
    
    def run_analysis(self, process_generator: ProcessGenerator, process_name: str) -> Dict:
        """
        Run complete IB analysis for a single stochastic process.
        """
        print(f"Analyzing process: {process_name}")
        
        # Generate sequence data
        sequence = process_generator(self.config.sample_length, seed=self.config.random_seed)
        
        # Extract empirical probabilities
        past_words, past_probabilities, future_conditional_probabilities = self.analysis_engine.extract_empirical_future_given_past(
            sequence, 
            past_window_length=self.config.past_window_length,
            future_window_length=self.config.future_window_length
        )
        
        print(f"  Extracted {len(past_words)} past contexts")
        
        # Run adaptive state growth
        growth_results = self._run_adaptive_state_growth(
            past_probabilities,
            future_conditional_probabilities
        )
        
        # Analyze phase transitions
        phase_analysis = self._compute_phase_transition_indicators(
            growth_results['beta_values'],
            growth_results['num_states'],
            growth_results['free_energies']
        )
        
        # Compile complete results
        results = {
            'process_name': process_name,
            'sequence_length': len(sequence),
            'empirical_mutual_information': self.analysis_engine.compute_empirical_mutual_information(
                past_probabilities, future_conditional_probabilities
            ),
            'beta_values': growth_results['beta_values'],
            'num_states': growth_results['num_states'],
            'free_energies': growth_results['free_energies'],
            'complexities': growth_results['complexities'],
            'accuracies': growth_results['accuracies'],
            'energies': growth_results['energies'],
            'entropies': growth_results['entropies'],
            'phase_transition_beta': phase_analysis.get('phase_transition_beta'),
            'critical_state_count': phase_analysis.get('critical_state_count'),
            'past_probabilities': past_probabilities,
            'future_conditional_probabilities': future_conditional_probabilities,
            'final_posterior': growth_results['final_posterior'],
            'final_emission': growth_results['final_emission']
        }
        
        # Store results
        self.results[process_name] = results
        print(f"  Analysis complete for {process_name}")
        
        return results
    
    def _run_adaptive_state_growth(self, past_probabilities, future_conditional_probabilities):
        """Run adaptive state growth across beta schedule."""
        results = {
            'beta_values': [],
            'num_states': [],
            'free_energies': [],
            'complexities': [],
            'accuracies': [],
            'energies': [],
            'entropies': [],
            'final_posterior': None,
            'final_emission': None
        }
        
        # Start with single state
        current_num_states = 1
        current_posterior = None
        current_emission = None
        
        for beta in self.beta_schedule:
            # Try state splitting if we haven't reached the maximum
            if current_num_states < self.config.maximum_states_allowed:
                if current_posterior is not None and current_emission is not None:
                    # Attempt to split states
                    new_posterior, new_emission, new_free_energy, split_occurred = self.optimization_engine.attempt_state_split(
                        past_probabilities,
                        future_conditional_probabilities,
                        beta,
                        current_posterior,
                        current_emission,
                        results['free_energies'][-1] if results['free_energies'] else float('inf')
                    )
                    
                    if split_occurred:
                        current_num_states = new_posterior.shape[1]
                        current_posterior = new_posterior
                        current_emission = new_emission
                        print(f"    Split to {current_num_states} states at β={beta:.3f}")
            
            # Run EM optimization with current number of states
            posterior, emission, free_energy, complexity, accuracy, energy, entropy = self.optimization_engine.run_em_coordinate_ascent(
                past_probabilities,
                future_conditional_probabilities,
                beta,
                initial_emission_probabilities=current_emission,
                initial_posterior_distribution=current_posterior
            )
            
            # Update current state
            current_posterior = posterior
            current_emission = emission
            
            # Store results
            results['beta_values'].append(beta)
            results['num_states'].append(current_num_states)
            results['free_energies'].append(free_energy)
            results['complexities'].append(complexity)
            results['accuracies'].append(accuracy)
            results['energies'].append(energy)
            results['entropies'].append(entropy)
        
        results['final_posterior'] = current_posterior
        results['final_emission'] = current_emission
        
        return results
    
    def _compute_phase_transition_indicators(self, beta_values, num_states, free_energies):
        """Compute phase transition indicators."""
        beta_array = np.array(beta_values)
        states_array = np.array(num_states)
        
        # Find first significant state increase
        state_changes = np.diff(states_array)
        significant_changes = np.where(state_changes > 0)[0]
        
        if len(significant_changes) > 0:
            transition_index = significant_changes[0]
            return {
                'phase_transition_beta': beta_array[transition_index],
                'critical_state_count': states_array[transition_index + 1]
            }
        
        return {}
    
    def run_analysis_for_multiple_processes(self, process_generators: Dict[str, ProcessGenerator]) -> Dict:
        """Run analysis for multiple processes."""
        all_results = {}
        
        for process_name, generator in process_generators.items():
            result = self.run_analysis(generator, process_name)
            all_results[process_name] = result
        
        self.analysis_complete = True
        return all_results
    
    def create_information_bottleneck_plot(self, results: Optional[Dict] = None):
        """Create information bottleneck trade-off plot using visualization module."""
        if results is None:
            results = self.results
        
        from visualization import create_information_bottleneck_plot
        return create_information_bottleneck_plot(results)
    
    def create_phase_transition_plot(self, results: Optional[Dict] = None):
        """Create phase transition plot using visualization module."""
        if results is None:
            results = self.results
        
        from visualization import create_phase_transition_plot
        return create_phase_transition_plot(results)
    
    def create_vfe_decomposition_plot(self, results: Optional[Dict] = None):
        """Create VFE decomposition plot showing both accuracy/complexity and energy/entropy."""
        if results is None:
            results = self.results
        
        from visualization import create_vfe_decomposition_plot
        return create_vfe_decomposition_plot(results)


# Convenience function for quick analysis
def run_default_analysis(process_generators: Optional[Dict] = None,
                        selected_processes: Optional[List[str]] = None):
    """
    Convenience function to run default analysis with standard implementations.
    """
    from interfaces import create_batch_analyzer
    
    # Create analyzer with simplified dependencies
    analyzer = create_batch_analyzer()
    
    # Get process generators
    if process_generators is None:
        from processes import PROCESS_GENERATORS
        process_generators = PROCESS_GENERATORS
    
    # Filter processes if specified
    if selected_processes is not None:
        process_generators = {
            name: gen for name, gen in process_generators.items() 
            if name in selected_processes
        }
    
    # Run analysis
    results = analyzer.run_analysis_for_multiple_processes(process_generators)
    
    return analyzer, results


if __name__ == "__main__":
    # Run demo analysis
    print("Running batch information bottleneck analysis...")
    analyzer, results = run_default_analysis(selected_processes=["Golden-Mean", "IID(p=0.5)"])
    
    # Create plots including VFE decomposition
    analyzer.create_information_bottleneck_plot()
    analyzer.create_phase_transition_plot()
    analyzer.create_vfe_decomposition_plot()
    
    from visualization import show_all_plots
    show_all_plots(save_to_dir="results/figures")
    
    print("Analysis complete!")