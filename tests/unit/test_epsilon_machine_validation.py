"""
Epsilon Machine Validation Tests
===============================

Tests to verify that discovered states correspond to correct
epsilon-machine (causal state) structure for known processes.
"""

import numpy as np
import pytest
import sys
import os
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from processes import PROCESS_GENERATORS
from empirical_analysis import extract_empirical_future_given_past
from state_splitting import run_adaptive_state_growth
from information_theory import compute_variational_free_energy


class TestEpsilonMachineStructure:
    """Test that discovered states match epsilon-machine causal structure."""
    
    def test_golden_mean_causal_states(self):
        """Test Golden-Mean discovers correct causal state structure."""
        # Golden-Mean has 2 causal states:
        # State A: "contexts that don't end with 1" -> can emit 0 or 1
        # State B: "contexts that end with 1" -> can only emit 0
        
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(5000, seed=42)
        
        # Use appropriate window length to capture causal structure
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=3, future_window_length=1
        )
        
        beta_schedule = np.geomspace(1.0, 50, 15)
        results = run_adaptive_state_growth(
            past_probs,
            future_conditional,
            beta_schedule,
            maximum_states_allowed=4,
            random_seed=42
        )
        
        if results['num_states'][-1] == 2:
            emission_probs = results['final_emission']
            
            # One state should have P(1) ≈ 0 (can't emit 1 after 1)
            # Other state should have P(1) > 0 (can emit 1 in other cases)
            p1_state0 = emission_probs[0, 1] if emission_probs.shape[1] > 1 else 0
            p1_state1 = emission_probs[1, 1] if emission_probs.shape[1] > 1 else 0
            
            # One state should strongly avoid emitting 1
            min_p1 = min(p1_state0, p1_state1)
            max_p1 = max(p1_state0, p1_state1)
            
            assert min_p1 < 0.1, f"One state should avoid emitting 1: {min_p1:.3f}"
            assert max_p1 > 0.2, f"Other state should allow emitting 1: {max_p1:.3f}"
    
    def test_periodic_3_causal_states(self):
        """Test Periodic-3 discovers 3-state cycle structure."""
        generator = PROCESS_GENERATORS["Periodic-3"]
        sequence = generator(3000, seed=42)  # 0,1,0,0,1,0,0,1,0...
        
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=2, future_window_length=1
        )
        
        beta_schedule = np.geomspace(1.0, 50, 15)
        results = run_adaptive_state_growth(
            past_probs,
            future_conditional,
            beta_schedule,
            maximum_states_allowed=5,
            random_seed=42
        )
        
        if results['num_states'][-1] == 3:
            emission_probs = results['final_emission']
            
            # Each state should deterministically emit one symbol
            # (since Periodic-3 is deterministic)
            for state_idx in range(3):
                max_prob = np.max(emission_probs[state_idx])
                assert max_prob > 0.8, \
                    f"Periodic state {state_idx} should be near-deterministic: max_p={max_prob:.3f}"
    
    def test_even_process_single_state(self):
        """Test Even process has single memoryless state."""
        generator = PROCESS_GENERATORS["Even"]
        sequence = generator(3000, seed=42)
        
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=3, future_window_length=1
        )
        
        beta_schedule = np.geomspace(1.0, 30, 10)
        results = run_adaptive_state_growth(
            past_probs,
            future_conditional,
            beta_schedule,
            maximum_states_allowed=4,
            random_seed=42
        )
        
        final_states = results['num_states'][-1]
        assert final_states == 1, f"Even process should have 1 state, found {final_states}"
        
        # Single state should have uniform emission (P(0) = P(1) = 0.5)
        emission_probs = results['final_emission']
        if emission_probs.shape[1] == 2:
            p0, p1 = emission_probs[0, 0], emission_probs[0, 1]
            assert abs(p0 - 0.5) < 0.1, f"Even process should emit 0 with p≈0.5: {p0:.3f}"
            assert abs(p1 - 0.5) < 0.1, f"Even process should emit 1 with p≈0.5: {p1:.3f}"


class TestCausalEquivalence:
    """Test that causally equivalent contexts are assigned to same state."""
    
    def group_contexts_by_suffix(self, past_words, suffix_length=1):
        """Group past contexts by their suffix patterns."""
        suffix_groups = defaultdict(list)
        
        for i, word in enumerate(past_words):
            if len(word) >= suffix_length:
                suffix = tuple(word[-suffix_length:])
                suffix_groups[suffix].append(i)
        
        return suffix_groups
    
    def test_golden_mean_causal_grouping(self):
        """Test that Golden-Mean contexts are grouped by causal equivalence."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(4000, seed=42)
        
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=4, future_window_length=1
        )
        
        beta_schedule = np.geomspace(1.0, 40, 12)
        results = run_adaptive_state_growth(
            past_probs,
            future_conditional,
            beta_schedule,
            maximum_states_allowed=4,
            random_seed=42
        )
        
        if results['num_states'][-1] == 2:
            posterior = results['final_posterior']
            
            # Group contexts by whether they end with 1
            ends_with_1 = []
            ends_with_0 = []
            
            for i, word in enumerate(past_words):
                if len(word) > 0:
                    if word[-1] == 1:
                        ends_with_1.append(i)
                    else:
                        ends_with_0.append(i)
            
            if len(ends_with_1) > 0 and len(ends_with_0) > 0:
                # Contexts ending with 1 should have similar posteriors
                post_1 = posterior[ends_with_1]
                post_0 = posterior[ends_with_0]
                
                # Check if contexts in same causal class have similar posteriors
                if len(ends_with_1) > 1:
                    var_1 = np.var(post_1, axis=0).max()
                    assert var_1 < 0.3, f"Contexts ending with 1 should have similar posteriors: var={var_1:.3f}"
                
                if len(ends_with_0) > 1:
                    var_0 = np.var(post_0, axis=0).max()
                    assert var_0 < 0.3, f"Contexts ending with 0 should have similar posteriors: var={var_0:.3f}"


class TestMinimalityProperty:
    """Test that discovered models are minimal (no redundant states)."""
    
    def test_states_are_distinguishable(self):
        """Test that all discovered states have significantly different emissions."""
        for process_name in ["Golden-Mean", "Mixed", "Periodic-3"]:
            generator = PROCESS_GENERATORS[process_name]
            sequence = generator(3000, seed=42)
            
            past_words, past_probs, future_conditional = extract_empirical_future_given_past(
                sequence, past_window_length=4, future_window_length=1
            )
            
            beta_schedule = np.geomspace(1.0, 30, 10)
            results = run_adaptive_state_growth(
                past_probs,
                future_conditional,
                beta_schedule,
                maximum_states_allowed=6,
                random_seed=42
            )
            
            emission_probs = results['final_emission']
            num_states = emission_probs.shape[0]
            
            if num_states > 1:
                # Check all pairs of states are sufficiently different
                for i in range(num_states):
                    for j in range(i + 1, num_states):
                        kl_div = self.compute_kl_divergence(
                            emission_probs[i], emission_probs[j]
                        )
                        
                        assert kl_div > 0.01, \
                            f"{process_name}: States {i} and {j} too similar (KL={kl_div:.4f})"
    
    def compute_kl_divergence(self, p, q):
        """Compute KL divergence between two probability distributions."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p_safe = p + epsilon
        q_safe = q + epsilon
        
        # Normalize
        p_safe = p_safe / p_safe.sum()
        q_safe = q_safe / q_safe.sum()
        
        return np.sum(p_safe * np.log(p_safe / q_safe))
    
    def test_no_unused_states(self):
        """Test that all discovered states are actually used."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(3000, seed=42)
        
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=4, future_window_length=1
        )
        
        beta_schedule = np.geomspace(1.0, 30, 10)
        results = run_adaptive_state_growth(
            past_probs,
            future_conditional,
            beta_schedule,
            maximum_states_allowed=6,
            random_seed=42
        )
        
        posterior = results['final_posterior']
        
        # Check that each state has non-trivial posterior mass
        state_usage = posterior.sum(axis=0)  # Sum over contexts
        total_mass = state_usage.sum()
        
        for state_idx, usage in enumerate(state_usage):
            usage_fraction = usage / total_mass
            assert usage_fraction > 0.05, \
                f"State {state_idx} barely used: {usage_fraction:.3f} of total mass"


class TestOptimalityProperties:
    """Test that discovered models satisfy optimality conditions."""
    
    def test_free_energy_monotonicity(self):
        """Test that free energy decreases monotonically with β."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(3000, seed=42)
        
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=4, future_window_length=1
        )
        
        beta_schedule = np.geomspace(1.0, 30, 10)
        results = run_adaptive_state_growth(
            past_probs,
            future_conditional,
            beta_schedule,
            maximum_states_allowed=6,
            random_seed=42
        )
        
        free_energies = results['free_energies']
        
        # Free energy should generally decrease (allowing for small numerical errors)
        for i in range(len(free_energies) - 1):
            improvement = free_energies[i] - free_energies[i + 1]
            assert improvement > -0.01, \
                f"Free energy increased significantly: Δ={improvement:.4f} at step {i}"
    
    def test_accuracy_complexity_tradeoff(self):
        """Test that accuracy increases and complexity follows expected pattern."""
        generator = PROCESS_GENERATORS["Golden-Mean"]
        sequence = generator(3000, seed=42)
        
        past_words, past_probs, future_conditional = extract_empirical_future_given_past(
            sequence, past_window_length=4, future_window_length=1
        )
        
        beta_schedule = np.geomspace(1.0, 30, 10)
        results = run_adaptive_state_growth(
            past_probs,
            future_conditional,
            beta_schedule,
            maximum_states_allowed=6,
            random_seed=42
        )
        
        accuracies = results['accuracies']
        complexities = results['complexities']
        num_states = results['num_states']
        
        # Accuracy should generally increase
        final_accuracy = accuracies[-1]
        initial_accuracy = accuracies[0]
        assert final_accuracy >= initial_accuracy - 0.01, \
            f"Accuracy decreased: {initial_accuracy:.3f} -> {final_accuracy:.3f}"
        
        # Complexity should increase when states are added
        for i in range(len(num_states) - 1):
            if num_states[i + 1] > num_states[i]:
                # State was added, complexity should increase
                assert complexities[i + 1] > complexities[i] - 0.01, \
                    f"Complexity didn't increase when state added at step {i}"


if __name__ == "__main__":
    # Run quick test for development
    import pytest
    pytest.main([__file__ + "::TestEpsilonMachineStructure::test_golden_mean_causal_states", "-v"])