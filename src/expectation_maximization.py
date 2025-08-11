"""
Expectation-Maximization Algorithm for Information Bottleneck
===========================================================

Implementation of coordinate ascent EM algorithm for solving the
information bottleneck optimization problem. Alternates between:

E-step: Update posterior q(state|past) given current emissions
M-step: Update emissions p(future|state) given current posterior

Converges to local optimum of the variational free energy.
"""

import numpy as np
from information_theory import compute_kl_divergence, compute_variational_free_energy


def run_em_coordinate_ascent(past_probabilities,
                           future_conditional_probabilities,
                           inverse_temperature_beta,
                           initial_emission_probabilities=None,
                           initial_posterior_distribution=None,
                           maximum_iterations=400,
                           convergence_tolerance=1e-6,
                           random_seed=42):
    """
    Run EM coordinate ascent to optimize information bottleneck objective.
    
    Alternates between updating posterior distribution (E-step) and 
    emission probabilities (M-step) until convergence.
    
    Args:
        past_probabilities: P(past) marginal distribution [N_past]
        future_conditional_probabilities: P(future|past) empirical [N_past, N_future]
        inverse_temperature_beta: Trade-off parameter β
        initial_emission_probabilities: Starting emissions [N_states, N_future] 
        initial_posterior_distribution: Starting posterior [N_past, N_states]
        maximum_iterations: Maximum number of EM iterations
        convergence_tolerance: Tolerance for emission change convergence
        random_seed: Random seed for initialization
        
    Returns:
        tuple: (posterior, emission, free_energy, complexity, accuracy)
    """
    rng = np.random.default_rng(random_seed)
    
    # Initialize number of states
    if initial_emission_probabilities is not None:
        num_states = initial_emission_probabilities.shape[0]
    else:
        num_states = 1
    
    alphabet_size = future_conditional_probabilities.shape[1]
    num_past_contexts = len(past_probabilities)
    
    # Initialize emission probabilities p(future|state)
    if initial_emission_probabilities is not None:
        emission_probabilities = initial_emission_probabilities.copy()
    else:
        emission_probabilities = rng.random((num_states, alphabet_size))
        emission_probabilities /= emission_probabilities.sum(axis=1, keepdims=True)
    
    # Initialize posterior distribution q(state|past)
    if initial_posterior_distribution is not None:
        posterior_distribution = initial_posterior_distribution.copy()
    else:
        posterior_distribution = np.full((num_past_contexts, num_states), 1.0 / num_states)
    
    # EM iteration loop
    for iteration in range(maximum_iterations):
        # ----------------------------------------
        # E-step: Update posterior q(state|past)
        # ----------------------------------------
        # Compute distances (KL divergences) between empirical and emission distributions
        distance_matrix = np.array([
            [compute_kl_divergence(future_conditional_probabilities[past_idx], 
                                 emission_probabilities[state_idx])
             for state_idx in range(num_states)]
            for past_idx in range(num_past_contexts)
        ])
        
        # Update posterior using exponential family form
        posterior_distribution = np.exp(-inverse_temperature_beta * distance_matrix)
        posterior_distribution /= posterior_distribution.sum(axis=1, keepdims=True)
        
        # ----------------------------------------
        # M-step: Update emissions p(future|state)  
        # ----------------------------------------
        # Compute weighted average of empirical distributions for each state
        weights = past_probabilities[:, None] * posterior_distribution  # [N_past, N_states]
        
        new_emission_probabilities = (
            weights[:, :, None] * future_conditional_probabilities[:, None, :]
        ).sum(axis=0)  # Sum over past contexts
        
        # Normalize to ensure valid probability distributions
        new_emission_probabilities /= new_emission_probabilities.sum(axis=1, keepdims=True)
        
        # ----------------------------------------
        # Check convergence
        # ----------------------------------------
        emission_change = np.max(np.abs(new_emission_probabilities - emission_probabilities))
        if emission_change < convergence_tolerance:
            emission_probabilities = new_emission_probabilities
            break
            
        emission_probabilities = new_emission_probabilities
    
    # Compute final free energy and components
    free_energy, complexity, accuracy, energy, entropy = compute_variational_free_energy(
        past_probabilities, 
        posterior_distribution, 
        emission_probabilities, 
        inverse_temperature_beta, 
        future_conditional_probabilities
    )
    
    return posterior_distribution, emission_probabilities, free_energy, complexity, accuracy, energy, entropy


def run_em_with_annealing(past_probabilities,
                        future_conditional_probabilities,  
                        beta_schedule,
                        maximum_iterations_per_beta=100,
                        convergence_tolerance=1e-6,
                        random_seed=42):
    """
    Run EM algorithm with deterministic annealing over β schedule.
    
    Gradually increases β (inverse temperature) to avoid local minima.
    Uses solution from previous β as warm start for next β.
    
    Args:
        past_probabilities: P(past) marginal distribution
        future_conditional_probabilities: P(future|past) empirical  
        beta_schedule: Array of β values (typically increasing)
        maximum_iterations_per_beta: Max EM iterations per β value
        convergence_tolerance: EM convergence tolerance
        random_seed: Random seed for initialization
        
    Returns:
        dict: Results for each β including posterior, emission, free_energy, etc.
    """
    results = {
        'beta_values': [],
        'posterior_distributions': [],
        'emission_probabilities': [],
        'free_energies': [],
        'complexities': [],
        'accuracies': [],
        'num_states': []
    }
    
    # Initialize for first β
    previous_posterior = None
    previous_emission = None
    
    for beta in beta_schedule:
        # Run EM for current β with warm start from previous solution
        posterior, emission, free_energy, complexity, accuracy = run_em_coordinate_ascent(
            past_probabilities,
            future_conditional_probabilities,
            beta,
            initial_emission_probabilities=previous_emission,
            initial_posterior_distribution=previous_posterior,
            maximum_iterations=maximum_iterations_per_beta,
            convergence_tolerance=convergence_tolerance,
            random_seed=random_seed
        )
        
        # Store results
        results['beta_values'].append(beta)
        results['posterior_distributions'].append(posterior.copy())
        results['emission_probabilities'].append(emission.copy())
        results['free_energies'].append(free_energy)
        results['complexities'].append(complexity)
        results['accuracies'].append(accuracy)
        results['num_states'].append(emission.shape[0])
        
        # Use as warm start for next β
        previous_posterior = posterior
        previous_emission = emission
    
    # Convert lists to arrays where appropriate
    results['beta_values'] = np.array(results['beta_values'])
    results['free_energies'] = np.array(results['free_energies'])
    results['complexities'] = np.array(results['complexities'])
    results['accuracies'] = np.array(results['accuracies'])
    results['num_states'] = np.array(results['num_states'])
    
    return results


def compute_em_convergence_diagnostics(emission_history, posterior_history):
    """
    Compute convergence diagnostics for EM algorithm.
    
    Args:
        emission_history: List of emission probability arrays over iterations
        posterior_history: List of posterior distribution arrays over iterations
        
    Returns:
        dict: Convergence diagnostics including parameter changes and likelihood
    """
    if len(emission_history) < 2:
        return {'insufficient_history': True}
    
    # Parameter change norms
    emission_changes = []
    posterior_changes = []
    
    for i in range(1, len(emission_history)):
        emission_change = np.linalg.norm(emission_history[i] - emission_history[i-1])
        posterior_change = np.linalg.norm(posterior_history[i] - posterior_history[i-1])
        
        emission_changes.append(emission_change)
        posterior_changes.append(posterior_change)
    
    return {
        'emission_parameter_changes': np.array(emission_changes),
        'posterior_parameter_changes': np.array(posterior_changes),
        'final_emission_change': emission_changes[-1] if emission_changes else 0,
        'final_posterior_change': posterior_changes[-1] if posterior_changes else 0,
        'converged': (emission_changes[-1] < 1e-6) if emission_changes else False
    }


def initialize_em_with_kmeans_clustering(future_conditional_probabilities, 
                                       num_initial_states,
                                       random_seed=42):
    """
    Initialize EM algorithm using k-means clustering of empirical distributions.
    
    Clusters the P(future|past) distributions to get initial emission probabilities.
    NOTE: Requires scikit-learn to be installed.
    
    Args:
        future_conditional_probabilities: P(future|past) empirical [N_past, N_future]
        num_initial_states: Number of states to initialize
        random_seed: Random seed for clustering
        
    Returns:
        tuple: (initial_emission_probabilities, initial_posterior_distribution)
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("K-means initialization requires scikit-learn. Install with: pip install scikit-learn")
    
    # Cluster the empirical distributions
    rng = np.random.default_rng(random_seed)
    kmeans = KMeans(n_clusters=num_initial_states, random_state=random_seed, n_init=10)
    cluster_assignments = kmeans.fit_predict(future_conditional_probabilities)
    
    # Use cluster centers as initial emissions
    initial_emission_probabilities = kmeans.cluster_centers_
    
    # Initialize posterior as one-hot based on cluster assignments
    num_past_contexts = len(future_conditional_probabilities)
    initial_posterior_distribution = np.zeros((num_past_contexts, num_initial_states))
    initial_posterior_distribution[np.arange(num_past_contexts), cluster_assignments] = 1.0
    
    # Add small amount of noise to avoid deterministic initialization
    noise = rng.normal(scale=0.01, size=initial_posterior_distribution.shape)
    initial_posterior_distribution += noise
    initial_posterior_distribution = np.abs(initial_posterior_distribution)
    initial_posterior_distribution /= initial_posterior_distribution.sum(axis=1, keepdims=True)
    
    return initial_emission_probabilities, initial_posterior_distribution


if __name__ == "__main__":
    # Demonstration with synthetic data
    from processes import PROCESS_GENERATORS
    from empirical_analysis import extract_empirical_future_given_past
    
    # Generate test data
    generator = PROCESS_GENERATORS["Golden-Mean"] 
    test_sequence = generator(5000, seed=42)
    
    # Extract empirical probabilities
    past_words, past_probs, future_conditional = extract_empirical_future_given_past(
        test_sequence, past_window_length=4, future_window_length=2)
    
    print(f"Data: {len(past_words)} past contexts, alphabet size {future_conditional.shape[1]}")
    
    # Run EM algorithm
    beta = 5.0
    posterior, emission, free_energy, complexity, accuracy = run_em_coordinate_ascent(
        past_probs, future_conditional, beta, random_seed=42)
    
    print(f"\nEM Results (β={beta}):")
    print(f"Free energy: {free_energy:.3f}")
    print(f"Complexity: {complexity:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Number of states: {emission.shape[0]}")
    print(f"Emission probabilities shape: {emission.shape}")
    
    # Test annealing
    print("\nTesting annealing...")
    beta_schedule = np.geomspace(0.1, 10, 10)
    annealing_results = run_em_with_annealing(
        past_probs, future_conditional, beta_schedule, random_seed=42)
    
    print(f"Annealing over {len(beta_schedule)} β values:")
    print(f"Final free energy: {annealing_results['free_energies'][-1]:.3f}")
    print(f"State counts: {annealing_results['num_states']}")