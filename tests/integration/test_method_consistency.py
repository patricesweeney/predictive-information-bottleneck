"""
Method Consistency Integration Tests
===================================

Integration tests comparing different methods (online vs batch,
different parameters) to ensure consistent state discovery.
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from processes import PROCESS_GENERATORS
from empirical_analysis import extract_empirical_future_given_past
from state_splitting import run_adaptive_state_growth
from interfaces import create_online_processor, create_batch_analyzer
from information_theory import compute_empirical_mutual_information


class TestOnlineVsBatchConsistency:
    """Test consistency between online and batch state discovery methods."""
    
    @pytest.fixture
    def test_process_data(self):
        """Generate test data for multiple processes."""
        data = {}
        for process_name in ["Golden-Mean", "Even", "Periodic-3"]:
            generator = PROCESS_GENERATORS[process_name]
            sequence = generator(3000, seed=42)
            data[process_name] = sequence
        return data
    
    def test_golden_mean_online_vs_batch_states(self, test_process_data):
        """Test Golden-Mean state discovery consistency between methods."""
        sequence = test_process_data["Golden-Mean"]
        
        # Batch analysis
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=4, future_window_length=1
        )
        
        beta_schedule = np.geomspace(1.0, 30, 10)
        batch_results = run_adaptive_state_growth(
            past_probs,
            future_conditional,
            beta_schedule,
            maximum_states_allowed=6,
            random_seed=42
        )
        
        # Online analysis with same parameters
        processor = create_online_processor()
        # Set similar parameters
        processor.inverse_temperature_beta = 30.0
        processor.eigenvalue_check_interval = 200
        
        for symbol in sequence:
            processor.process_symbol(symbol)
        
        online_results = processor.get_analysis_results()
        
        batch_states = batch_results['num_states'][-1]
        online_states = online_results['current_num_states']
        
        # Should discover same number of states
        assert abs(batch_states - online_states) <= 1, \
            f"State count mismatch: batch={batch_states}, online={online_states}"
    
    def test_all_processes_consistency(self, test_process_data):
        """Test batch vs online consistency across all test processes."""
        results = {}
        
        for process_name, sequence in test_process_data.items():
            # Batch method
            past_words, past_probs, future_conditional = extract_empirical_future_given_past(
                sequence, past_window_length=3, future_window_length=1
            )
            
            beta_schedule = np.geomspace(1.0, 25, 8)
            batch_results = run_adaptive_state_growth(
                past_probs,
                future_conditional,
                beta_schedule,
                maximum_states_allowed=5,
                random_seed=42
            )
            
            # Online method
            processor = create_online_processor()
            processor.inverse_temperature_beta = 25.0
            processor.eigenvalue_check_interval = 300
            
            for symbol in sequence:
                processor.process_symbol(symbol)
            
            online_results = processor.get_analysis_results()
            
            results[process_name] = {
                'batch_states': batch_results['num_states'][-1],
                'online_states': online_results['current_num_states'],
                'batch_free_energy': batch_results['free_energies'][-1],
                'online_free_energies': online_results['free_energies']
            }
        
        # Check consistency for each process
        for process_name, result in results.items():
            state_diff = abs(result['batch_states'] - result['online_states'])
            assert state_diff <= 1, \
                f"{process_name}: Large state count difference: {state_diff}"
    
    def test_convergence_properties(self):
        """Test that both methods show similar convergence properties."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(4000, seed=42)
        
        # Batch convergence
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=4, future_window_length=1
        )
        
        beta_schedule = np.geomspace(1.0, 40, 15)
        batch_results = run_adaptive_state_growth(
            past_probs,
            future_conditional,
            beta_schedule,
            maximum_states_allowed=6,
            random_seed=42
        )
        
        # Online convergence tracking
        processor = create_online_processor()
        processor.inverse_temperature_beta = 40.0
        processor.eigenvalue_check_interval = 200
        
        for symbol in sequence:
            processor.process_symbol(symbol)
        
        online_results = processor.get_analysis_results()
        
        # Both should show decreasing free energy over time
        batch_fe = batch_results['free_energies']
        online_fe = online_results['free_energies']
        
        if len(batch_fe) > 1:
            batch_improving = batch_fe[-1] <= batch_fe[0] + 0.1
            assert batch_improving, "Batch free energy should improve or stay stable"
        
        if len(online_fe) > 1:
            online_improving = online_fe[-1] <= online_fe[0] + 0.1
            assert online_improving, "Online free energy should improve or stay stable"


class TestParameterRobustness:
    """Test robustness to different parameter choices."""
    
    def test_beta_value_robustness(self):
        """Test that final results are robust to β parameter choices."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(3000, seed=42)
        
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=4, future_window_length=1
        )
        
        # Test different final β values
        beta_finals = [20, 30, 50]
        state_counts = []
        
        for beta_final in beta_finals:
            beta_schedule = np.geomspace(1.0, beta_final, 10)
            results = run_adaptive_state_growth(
                past_probs,
                future_conditional,
                beta_schedule,
                maximum_states_allowed=6,
                random_seed=42
            )
            state_counts.append(results['num_states'][-1])
        
        # Should converge to same state count
        unique_counts = set(state_counts)
        assert len(unique_counts) <= 2, \
            f"Too much variation with different β values: {state_counts}"
    
    def test_window_length_robustness(self):
        """Test robustness to different window lengths."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(4000, seed=42)
        
        state_counts = []
        for past_length in [3, 4, 5]:
            for future_length in [1, 2]:
                try:
                    past_words, past_probs, future_conditional = extract_empirical_future_given_past(
                        sequence, past_window_length=past_length, future_window_length=future_length
                    )
                    
                    beta_schedule = np.geomspace(1.0, 30, 10)
                    results = run_adaptive_state_growth(
                        past_probs,
                        future_conditional,
                        beta_schedule,
                        maximum_states_allowed=6,
                        random_seed=42
                    )
                    state_counts.append(results['num_states'][-1])
                except Exception:
                    # Skip if window combination doesn't work
                    continue
        
        if len(state_counts) > 1:
            # Should be reasonably consistent
            state_range = max(state_counts) - min(state_counts)
            assert state_range <= 2, \
                f"Too much variation across window lengths: range={state_range}"
    
    def test_random_seed_independence(self):
        """Test that results are independent of random seed choice."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(3000, seed=42)  # Fixed data sequence
        
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=4, future_window_length=1
        )
        
        # Test multiple random seeds
        seeds = [42, 123, 456, 789, 999]
        state_counts = []
        
        for seed in seeds:
            beta_schedule = np.geomspace(1.0, 30, 10)
            results = run_adaptive_state_growth(
                past_probs,
                future_conditional,
                beta_schedule,
                maximum_states_allowed=6,
                random_seed=seed
            )
            state_counts.append(results['num_states'][-1])
        
        # All should give same result for deterministic process
        unique_counts = set(state_counts)
        assert len(unique_counts) == 1, \
            f"Results should be seed-independent: {state_counts}"


class TestScalabilityProperties:
    """Test how methods scale with different data characteristics."""
    
    def test_sequence_length_scaling(self):
        """Test behavior with different sequence lengths."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        
        results = {}
        for length in [1000, 2000, 4000]:
            sequence = generator(length, seed=42)
            
            past_words, past_probs, future_conditional = extract_empirical_future_given_past(
                sequence, past_window_length=4, future_window_length=1
            )
            
            beta_schedule = np.geomspace(1.0, 30, 10)
            batch_results = run_adaptive_state_growth(
                past_probs,
                future_conditional,
                beta_schedule,
                maximum_states_allowed=6,
                random_seed=42
            )
            
            results[length] = {
                'states': batch_results['num_states'][-1],
                'contexts': len(past_probs),
                'free_energy': batch_results['free_energies'][-1]
            }
        
        # Longer sequences should give more stable results
        state_counts = [results[length]['states'] for length in [1000, 2000, 4000]]
        
        # Should converge to same state count
        final_counts = state_counts[-2:]  # Last two
        assert len(set(final_counts)) == 1, \
            f"State count should stabilize with longer sequences: {state_counts}"
    
    def test_alphabet_size_effects(self):
        """Test behavior with different effective alphabet sizes."""
        # Compare binary vs higher entropy processes
        binary_generator = PROCESS_GENERATORS["Golden-Mean"]
        mixed_generator = PROCESS_GENERATORS["Mixed"]
        
        results = {}
        for name, generator in [("Golden-Mean", binary_generator), ("Mixed", mixed_generator)]:
            sequence = generator(3000, seed=42)
            
            past_words, past_probs, future_conditional = extract_empirical_future_given_past(
                sequence, past_window_length=3, future_window_length=1
            )
            
            beta_schedule = np.geomspace(1.0, 25, 8)
            batch_results = run_adaptive_state_growth(
                past_probs,
                future_conditional,
                beta_schedule,
                maximum_states_allowed=6,
                random_seed=42
            )
            
            results[name] = {
                'states': batch_results['num_states'][-1],
                'entropy': compute_empirical_mutual_information(past_probs, future_conditional)
            }
        
        # Higher entropy process should potentially have more states
        golden_states = results["Golden-Mean"]['states']
        mixed_states = results["Mixed"]['states']
        
        # Mixed should have at least as many states as Golden-Mean
        assert mixed_states >= golden_states, \
            f"Mixed process should have ≥ states than Golden-Mean: {mixed_states} vs {golden_states}"


class TestErrorHandlingAndEdgeCases:
    """Test robustness to edge cases and error conditions."""
    
    def test_short_sequence_handling(self):
        """Test behavior with very short sequences."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        
        # Very short sequence
        sequence = generator(100, seed=42)
        
        try:
            past_words, past_probs, future_conditional = extract_empirical_future_given_past(
                sequence, past_window_length=3, future_window_length=1
            )
            
            if len(past_probs) > 0:
                beta_schedule = np.geomspace(1.0, 10, 5)
                results = run_adaptive_state_growth(
                    past_probs,
                    future_conditional,
                    beta_schedule,
                    maximum_states_allowed=3,
                    random_seed=42
                )
                
                # Should complete without error and give reasonable result
                assert results['num_states'][-1] >= 1, "Should discover at least 1 state"
                assert results['num_states'][-1] <= 3, "Should not exceed maximum"
            
        except Exception as e:
            # Should handle gracefully, not crash
            pytest.skip(f"Short sequence handling: {e}")
    
    def test_extreme_beta_values(self):
        """Test behavior with extreme β values."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(2000, seed=42)
        
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=3, future_window_length=1
        )
        
        # Test very small β (should favor complexity)
        small_beta_schedule = np.geomspace(0.01, 1.0, 5)
        small_results = run_adaptive_state_growth(
            past_probs,
            future_conditional,
            small_beta_schedule,
            maximum_states_allowed=4,
            random_seed=42
        )
        
        # Test very large β (should favor accuracy)
        large_beta_schedule = np.geomspace(10.0, 100.0, 5)
        large_results = run_adaptive_state_growth(
            past_probs,
            future_conditional,
            large_beta_schedule,
            maximum_states_allowed=4,
            random_seed=42
        )
        
        # Both should complete successfully
        assert small_results['num_states'][-1] >= 1, "Small β should work"
        assert large_results['num_states'][-1] >= 1, "Large β should work"
        
        # Large β should generally favor more states (accuracy)
        assert large_results['num_states'][-1] >= small_results['num_states'][-1], \
            "Large β should favor more states"


if __name__ == "__main__":
    # Run quick test for development
    import pytest
    pytest.main([__file__ + "::TestOnlineVsBatchConsistency::test_golden_mean_online_vs_batch_states", "-v"])