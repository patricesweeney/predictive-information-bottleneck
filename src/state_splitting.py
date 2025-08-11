"""
State Splitting for Information Bottleneck
=========================================

Implementation of eigenvalue-based automatic state splitting following
Still et al. (2014). Monitors linear stability of states and splits
unstable states to grow the model complexity adaptively.

Key insight: When β * λ_max > 1 for a state's Hessian eigenvalue,
that state becomes linearly unstable and should be split.
"""

import numpy as np
from information_theory import (compute_hessian_eigenvalue_for_state, 
                              compute_variational_free_energy)
from expectation_maximization import run_em_coordinate_ascent


def analyze_state_stability(past_probabilities,
                          posterior_distribution,
                          emission_probabilities,
                          future_conditional_probabilities,
                          inverse_temperature_beta):
    """
    Analyze linear stability of all states using Hessian eigenvalue analysis.
    
    Computes the largest eigenvalue for each state's Hessian matrix.
    States with β * λ_max > 1 are linearly unstable and candidates for splitting.
    
    Args:
        past_probabilities: P(past) marginal distribution [N_past]
        posterior_distribution: q(state|past) [N_past, N_states]
        emission_probabilities: p(future|state) [N_states, N_future]
        future_conditional_probabilities: P(future|past) [N_past, N_future]
        inverse_temperature_beta: Trade-off parameter β
        
    Returns:
        dict: Stability analysis results
    """
    num_states = emission_probabilities.shape[0]
    
    # Compute eigenvalues for all states
    eigenvalues = []
    stability_criteria = []
    
    for state_index in range(num_states):
        max_eigenvalue = compute_hessian_eigenvalue_for_state(
            state_index, 
            past_probabilities, 
            posterior_distribution,
            emission_probabilities, 
            future_conditional_probabilities
        )
        
        eigenvalues.append(max_eigenvalue)
        stability_criteria.append(inverse_temperature_beta * max_eigenvalue)
    
    # Find most unstable state
    stability_criteria = np.array(stability_criteria)
    most_unstable_state = int(np.argmax(stability_criteria))
    max_stability_criterion = stability_criteria[most_unstable_state]
    
    return {
        'eigenvalues': np.array(eigenvalues),
        'stability_criteria': stability_criteria,
        'most_unstable_state': most_unstable_state,
        'max_stability_criterion': max_stability_criterion,
        'any_unstable': max_stability_criterion > 1.0,
        'instability_threshold': 1.0
    }


def attempt_state_split(past_probabilities,
                       future_conditional_probabilities,
                       inverse_temperature_beta,
                       current_posterior_distribution,
                       current_emission_probabilities,
                       current_free_energy,
                       minimum_free_energy_improvement=0.01,
                       maximum_states_allowed=16,
                       perturbation_scale=1e-2,
                       random_seed=42):
    """
    Attempt to split the most unstable state if conditions are met.
    
    Implements the Still et al. (2014) state splitting criterion:
    1. Find state with largest β * λ_max
    2. Check if β * λ_max > 1 (linear instability) 
    3. If unstable, split the state and verify free energy improves
    
    Args:
        past_probabilities: P(past) marginal distribution
        future_conditional_probabilities: P(future|past) empirical
        inverse_temperature_beta: Trade-off parameter β
        current_posterior_distribution: Current q(state|past)
        current_emission_probabilities: Current p(future|state)
        current_free_energy: Current free energy value
        minimum_free_energy_improvement: Required ΔF for accepting split
        maximum_states_allowed: Maximum number of states to allow
        perturbation_scale: Scale of random perturbation for splitting
        random_seed: Random seed for perturbation
        
    Returns:
        tuple: (new_posterior, new_emission, new_free_energy, split_occurred)
    """
    current_num_states = current_emission_probabilities.shape[0]
    
    # Check if we've reached maximum states
    if current_num_states >= maximum_states_allowed:
        return (current_posterior_distribution, 
                current_emission_probabilities, 
                current_free_energy, 
                False)
    
    # Analyze state stability
    stability_analysis = analyze_state_stability(
        past_probabilities,
        current_posterior_distribution,
        current_emission_probabilities,
        future_conditional_probabilities,
        inverse_temperature_beta
    )
    
    # Check instability criterion
    if not stability_analysis['any_unstable']:
        return (current_posterior_distribution, 
                current_emission_probabilities, 
                current_free_energy, 
                False)
    
    # Proceed with splitting the most unstable state
    state_to_split = stability_analysis['most_unstable_state']
    
    # Create new emission probabilities by duplicating and perturbing
    new_emission_probabilities = create_split_emission_probabilities(
        current_emission_probabilities, 
        state_to_split, 
        perturbation_scale, 
        random_seed
    )
    
    # Create new posterior distribution
    new_posterior_distribution = create_split_posterior_distribution(
        current_posterior_distribution, 
        state_to_split
    )
    
    # Optimize the split configuration using EM
    optimized_posterior, optimized_emission, new_free_energy, _, _, _, _ = run_em_coordinate_ascent(
        past_probabilities,
        future_conditional_probabilities,
        inverse_temperature_beta,
        initial_emission_probabilities=new_emission_probabilities,
        initial_posterior_distribution=new_posterior_distribution,
        random_seed=random_seed
    )
    
    # Check if free energy improved sufficiently
    free_energy_improvement = current_free_energy - new_free_energy
    if free_energy_improvement > minimum_free_energy_improvement:
        return (optimized_posterior, optimized_emission, new_free_energy, True)
    else:
        return (current_posterior_distribution, 
                current_emission_probabilities, 
                current_free_energy, 
                False)


def create_split_emission_probabilities(current_emission_probabilities, 
                                       state_to_split_index, 
                                       perturbation_scale=1e-2,
                                       random_seed=42):
    """
    Create new emission probability matrix by splitting one state.
    
    Duplicates the emission vector for the target state and adds
    random perturbation to break symmetry.
    
    Args:
        current_emission_probabilities: Current p(future|state) [N_states, N_future]
        state_to_split_index: Index of state to split
        perturbation_scale: Standard deviation of perturbation
        random_seed: Random seed for perturbation
        
    Returns:
        np.array: New emission probabilities [N_states+1, N_future]
    """
    rng = np.random.default_rng(random_seed)
    
    # Get emission vector for state to split
    original_emission_vector = current_emission_probabilities[state_to_split_index]
    
    # Create perturbed copy
    perturbation = rng.normal(scale=perturbation_scale, size=original_emission_vector.shape)
    perturbed_emission_vector = original_emission_vector + perturbation
    
    # Ensure non-negativity and normalization
    perturbed_emission_vector = np.clip(perturbed_emission_vector, 1e-6, None)
    perturbed_emission_vector /= perturbed_emission_vector.sum()
    
    # Stack original emissions with new perturbed state
    new_emission_probabilities = np.vstack([
        current_emission_probabilities,
        perturbed_emission_vector[None, :]
    ])
    
    return new_emission_probabilities


def create_split_posterior_distribution(current_posterior_distribution, 
                                       state_to_split_index):
    """
    Create new posterior distribution by splitting one state.
    
    Splits the posterior probability mass for the target state
    equally between the original and new state.
    
    Args:
        current_posterior_distribution: Current q(state|past) [N_past, N_states]
        state_to_split_index: Index of state to split
        
    Returns:
        np.array: New posterior distribution [N_past, N_states+1]
    """
    # Add column for new state (initialized to zero)
    new_posterior_distribution = np.pad(
        current_posterior_distribution, 
        ((0, 0), (0, 1)), 
        mode='constant', 
        constant_values=0
    )
    
    # Split probability mass equally between original and new state
    original_mass = new_posterior_distribution[:, state_to_split_index].copy()
    new_posterior_distribution[:, state_to_split_index] = original_mass * 0.5
    new_posterior_distribution[:, -1] = original_mass * 0.5  # New state gets other half
    
    # Renormalize to ensure valid probability distributions
    new_posterior_distribution /= new_posterior_distribution.sum(axis=1, keepdims=True)
    
    return new_posterior_distribution


def run_adaptive_state_growth(past_probabilities,
                            future_conditional_probabilities,
                            beta_schedule,
                            initial_states=1,
                            maximum_states_allowed=16,
                            minimum_free_energy_improvement=0.01,
                            em_max_iterations=400,
                            random_seed=42):
    """
    Run information bottleneck with adaptive state growth over β schedule.
    
    For each β value:
    1. Run EM to convergence
    2. Check for unstable states
    3. Split unstable states if beneficial
    4. Repeat until no more beneficial splits
    
    Args:
        past_probabilities: P(past) marginal distribution
        future_conditional_probabilities: P(future|past) empirical
        beta_schedule: Array of β values for annealing
        initial_states: Number of states to start with
        maximum_states_allowed: Maximum states to allow
        minimum_free_energy_improvement: Required ΔF for splits
        em_max_iterations: Max EM iterations per β
        random_seed: Random seed
        
    Returns:
        dict: Results including states, free energies, split history
    """
    results = {
        'beta_values': [],
        'num_states': [],
        'free_energies': [],
        'complexities': [],
        'accuracies': [],
        'split_history': [],
        'final_posterior': None,
        'final_emission': None
    }
    
    # Initialize with single state or specified number
    rng = np.random.default_rng(random_seed)
    alphabet_size = future_conditional_probabilities.shape[1]
    num_past_contexts = len(past_probabilities)
    
    if initial_states == 1:
        emission_probabilities = np.full((1, alphabet_size), 1.0 / alphabet_size)
        posterior_distribution = np.full((num_past_contexts, 1), 1.0)
    else:
        emission_probabilities = rng.random((initial_states, alphabet_size))
        emission_probabilities /= emission_probabilities.sum(axis=1, keepdims=True)
        posterior_distribution = np.full((num_past_contexts, initial_states), 1.0 / initial_states)
    
    # Process each β value
    for beta_index, beta in enumerate(beta_schedule):
        # Run EM to convergence for current β
        posterior_distribution, emission_probabilities, free_energy, complexity, accuracy = run_em_coordinate_ascent(
            past_probabilities,
            future_conditional_probabilities,
            beta,
            initial_emission_probabilities=emission_probabilities,
            initial_posterior_distribution=posterior_distribution,
            maximum_iterations=em_max_iterations,
            random_seed=random_seed
        )
        
        # Attempt state splits until no more beneficial splits
        splits_this_beta = 0
        while True:
            posterior_distribution, emission_probabilities, free_energy, split_occurred = attempt_state_split(
                past_probabilities,
                future_conditional_probabilities,
                beta,
                posterior_distribution,
                emission_probabilities,
                free_energy,
                minimum_free_energy_improvement,
                maximum_states_allowed,
                random_seed=random_seed + beta_index + splits_this_beta
            )
            
            if not split_occurred:
                break
            splits_this_beta += 1
        
        # Store results
        results['beta_values'].append(beta)
        results['num_states'].append(emission_probabilities.shape[0])
        results['free_energies'].append(free_energy)
        results['complexities'].append(complexity)
        results['accuracies'].append(accuracy)
        results['split_history'].append(splits_this_beta)
    
    # Convert to arrays and store final state
    results['beta_values'] = np.array(results['beta_values'])
    results['num_states'] = np.array(results['num_states'])
    results['free_energies'] = np.array(results['free_energies'])
    results['complexities'] = np.array(results['complexities'])
    results['accuracies'] = np.array(results['accuracies'])
    results['final_posterior'] = posterior_distribution
    results['final_emission'] = emission_probabilities
    
    return results


if __name__ == "__main__":
    # Demonstration with synthetic data
    from processes import PROCESS_GENERATORS
    from empirical_analysis import extract_empirical_future_given_past
    
    # Generate test data from Golden Mean process
    generator = PROCESS_GENERATORS["Golden-Mean"]
    test_sequence = generator(10000, seed=42)
    
    # Extract empirical probabilities
    past_words, past_probs, future_conditional = extract_empirical_future_given_past(
        test_sequence, past_window_length=6, future_window_length=2)
    
    print(f"Test data: {len(past_words)} past contexts")
    
    # Test adaptive state growth
    beta_schedule = np.geomspace(0.1, 20, 20)
    
    results = run_adaptive_state_growth(
        past_probs,
        future_conditional,
        beta_schedule,
        maximum_states_allowed=8,
        random_seed=42
    )
    
    print(f"\nAdaptive state growth results:")
    print(f"β range: {beta_schedule[0]:.2f} to {beta_schedule[-1]:.2f}")
    print(f"State count progression: {results['num_states']}")
    print(f"Total splits per β: {results['split_history']}")
    print(f"Final free energy: {results['free_energies'][-1]:.3f}")
    print(f"Final number of states: {results['num_states'][-1]}")
    
    # Analyze final state stability
    final_stability = analyze_state_stability(
        past_probs,
        results['final_posterior'],
        results['final_emission'],
        future_conditional,
        beta_schedule[-1]
    )
    print(f"\nFinal state stability analysis:")
    print(f"Stability criteria (β*λ): {final_stability['stability_criteria']}")
    print(f"Any unstable states: {final_stability['any_unstable']}")