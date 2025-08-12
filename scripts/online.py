"""
Online Recursive Information Bottleneck Processing
=================================================

Implementation of the online Recursive Information Bottleneck (RIB) algorithm.
Converges to ε-machine predictive state partition as λ decreases.

Uses the new RIB class that implements the proper fixed-point equations.
"""

from typing import Dict, Any
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from interfaces import OnlineProcessor, AnalysisEngine, OptimizationEngine, AnalysisConfig
from recursive_information_bottleneck import RIB


class OnlineProcessorImpl(OnlineProcessor):
    """
    Online Recursive Information Bottleneck processor.
    
    Wraps the RIB implementation to conform to the existing interface
    while providing proper RIB functionality.
    """
    
    def __init__(self,
                 analysis_engine: AnalysisEngine,
                 optimization_engine: OptimizationEngine,
                 config: AnalysisConfig = None,
                 alphabet_size: int = 2,
                 num_states: int = 4,
                 tau_F: int = 1,
                 alpha: float = 1e-3,
                 update_interval: int = 50):
        """
        Initialize online RIB processor.
        
        Args:
            analysis_engine: Analysis engine (for compatibility)
            optimization_engine: Optimization engine (for compatibility)  
            config: Configuration (for compatibility)
            alphabet_size: Size of alphabet X
            num_states: Number of states S (fixed for run)
            tau_F: Length of future block
            alpha: Dirichlet smoothing parameter
            update_interval: Steps between reestimate calls
        """
        super().__init__(analysis_engine, optimization_engine, config)
        
        # Initialize RIB algorithm
        self.rib = RIB(
            alphabet_size=alphabet_size,
            num_states=num_states,
            tau_F=tau_F,
            alpha=alpha,
            update_interval=update_interval,
            seed=self.config.random_seed
        )
        
        # Compatibility fields
        self.alphabet_size = alphabet_size
        self.num_states = num_states
        
        # Tracking for analysis
        self.sequence_log = []
        self.state_log = []
        self.objective_log = []
    
    def reset(self):
        """Reset online learning state."""
        # Reset the underlying RIB algorithm
        self.rib = RIB(
            alphabet_size=self.alphabet_size,
            num_states=self.num_states,
            tau_F=self.rib.tau_F,
            alpha=self.rib.alpha,
            update_interval=self.rib.update_interval,
            seed=self.config.random_seed
        )
        
        # Clear logs
        self.sequence_log = []
        self.state_log = []
        self.objective_log = []
    
    def process_symbol(self, new_symbol: int, run_log=None):
        """Process a single new symbol using RIB algorithm."""
        # Log the symbol
        self.sequence_log.append(new_symbol)
        
        # Process through RIB with optional run logging
        state = self.rib.partial_fit(new_symbol, run_log=run_log)
        
        # Log the assigned state
        self.state_log.append(state)
        
        # Log objective every 100 steps
        if len(self.sequence_log) % 100 == 0:
            obj = self.rib.objective()
            self.objective_log.append(obj)
        
        return state
    
    def get_rib_analysis_results(self) -> Dict[str, Any]:
        """Get RIB-specific analysis results."""
        return {
            'sequence_length': len(self.sequence_log),
            'unique_states': len(set(self.state_log)) if self.state_log else 0,
            'active_states': self.rib._count_active_states(),
            'current_lambda': self.rib._get_current_lambda(),
            'objective_history': self.objective_log,
            'state_sequence': self.state_log[-100:],  # Last 100 states
            'debug_info': self.rib.get_debug_info()
        }
    
    def get_predictive_distributions(self) -> Dict[int, Dict]:
        """Get predictive distributions for all states."""
        result = {}
        for s in range(self.rib.num_states):
            try:
                result[s] = self.rib.predictive_dist(s)
            except Exception:
                result[s] = {}
        return result
    
    def get_analysis_results(self):
        """Get current analysis results (compatibility method)."""
        rib_results = self.get_rib_analysis_results()
        
        # Convert to old format for compatibility
        return {
            'time_steps': list(range(0, len(self.objective_log) * 100, 100)),
            'free_energies': [obj.get('J', 0) for obj in self.objective_log],
            'complexities': [obj.get('I_s_past', 0) for obj in self.objective_log],
            'accuracies': [obj.get('I_s_future', 0) for obj in self.objective_log],
            'state_counts': [rib_results['active_states']] * len(self.objective_log),
            'split_times': [],  # RIB doesn't do dynamic splitting
            'current_num_states': rib_results['active_states'],
            'total_contexts': rib_results['sequence_length']
        }
    
    def create_online_analysis_plot(self):
        """Create plots showing online learning progress using visualization module."""
        results = self.get_analysis_results()
        
        from visualization import create_online_analysis_plot
        return create_online_analysis_plot(results)


# Convenience function for RIB analysis
def run_rib_analysis(process_generator, 
                     sequence_length: int = 10000,
                     process_name: str = "Unknown",
                     alphabet_size: int = 2,
                     num_states: int = 4,
                     tau_F: int = 1,
                     create_report: bool = False,
                     outdir: str = "results/rib_reports"):
    """
    Convenience function to run RIB analysis with optional visualization.
    """
    
    # Create processor with RIB parameters
    analysis_engine = None
    optimization_engine = None
    config = AnalysisConfig(random_seed=42)
    
    processor = OnlineProcessorImpl(
        analysis_engine=analysis_engine,
        optimization_engine=optimization_engine,
        config=config,
        alphabet_size=alphabet_size,
        num_states=num_states,
        tau_F=tau_F
    )
    
    # Create run log for visualization
    run_log = None
    if create_report:
        from rib_plots import RIBRunLog
        run_log = RIBRunLog()
    
    # Generate sequence and process
    sequence = process_generator(sequence_length, seed=42)
    
    print(f"Processing {sequence_length} symbols from {process_name} with RIB...")
    print(f"Parameters: alphabet_size={alphabet_size}, num_states={num_states}, tau_F={tau_F}")
    
    for i, symbol in enumerate(sequence):
        state = processor.process_symbol(symbol, run_log=run_log)
        
        # Print progress
        if i % 1000 == 0 and i > 0:
            rib_results = processor.get_rib_analysis_results()
            print(f"Step {i}: λ={rib_results['current_lambda']:.4f}, "
                  f"active_states={rib_results['active_states']}, "
                  f"current_state={state}")
    
    # Final checkpoint
    if run_log is not None:
        final_lambda = processor.rib._get_current_lambda()
        run_log.log_checkpoint(processor.rib.step_count, processor.rib, 
                             final_lambda, "final")
    
    # Final results
    rib_results = processor.get_rib_analysis_results()
    print("\nFinal RIB model:")
    print(f"  Active states: {rib_results['active_states']}/{num_states}")
    print(f"  Final λ: {rib_results['current_lambda']:.4f}")
    print(f"  Sequence length: {rib_results['sequence_length']}")
    
    # Show final objective
    if rib_results['objective_history']:
        final_obj = rib_results['objective_history'][-1]
        print(f"  Final objective: J={final_obj['J']:.4f}")
        print(f"  I(s;future)={final_obj['I_s_future']:.4f}")
        print(f"  I(s;past)={final_obj['I_s_past']:.4f}")
    
    # Generate visualization report
    if create_report and run_log is not None:
        print(f"\nGenerating visualization report...")
        from rib_plots import make_report
        make_report(run_log, outdir, process_name)
        print(f"Report saved to {outdir}")
    
    return processor, rib_results, run_log


# Backwards compatibility
def run_online_analysis(process_generator, 
                       sequence_length: int = 10000,
                       process_name: str = "Unknown"):
    """Backwards compatibility wrapper."""
    return run_rib_analysis(process_generator, sequence_length, process_name)


if __name__ == "__main__":
    # Run demo RIB analysis with visualization
    from processes import PROCESS_GENERATORS
    
    print("Running Recursive Information Bottleneck (RIB) analysis...")
    processor, results, run_log = run_rib_analysis(
        PROCESS_GENERATORS["Golden-Mean"], 
        sequence_length=3000,
        process_name="Golden-Mean",
        alphabet_size=2,
        num_states=4,
        tau_F=1,
        create_report=True,
        outdir="results/rib_reports/golden_mean"
    )
    
    print("\nPredictive distributions:")
    pred_dists = processor.get_predictive_distributions()
    for state, dist in pred_dists.items():
        if dist:  # Only show non-empty distributions
            print(f"  State {state}: {dist}")
    
    print("\nRIB analysis complete!")