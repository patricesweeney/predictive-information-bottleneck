"""
Stochastic Process Generators for Information Bottleneck Analysis
===============================================================

A collection of stochastic process generators with varying memory complexity:
- Zero memory: IID processes
- Finite memory: Markov chains and sofic processes  
- Infinite memory: Renewal and deterministic processes

Each generator returns a function that produces sequences given length and seed.
Statistical complexity (Cμ) ranges from 0 bits to infinite.
"""

import numpy as np


def create_random_number_generator(seed):
    """Convenience function for seeded RNG creation."""
    return np.random.default_rng(seed)


def create_iid_process(probability_of_one=0.5):
    """
    Factory for Independent Identically Distributed (IID) coin flip process.
    
    Memory complexity: Cμ = 0 bits (memoryless)
    
    Args:
        probability_of_one: Probability of emitting symbol '1'
        
    Returns:
        Generator function that produces IID sequences
    """
    def generate_sequence(length, *, seed=0):
        rng = create_random_number_generator(seed)
        probabilities = [1 - probability_of_one, probability_of_one]
        return rng.choice([0, 1], size=length, p=probabilities)
    
    generate_sequence.__name__ = f"IID(p={probability_of_one})"
    return generate_sequence


def create_two_state_markov_process(transition_prob_01=0.2, transition_prob_10=0.8):
    """
    Factory for two-state Markov chain process.
    
    Transition structure:
        State 0 --p01--> State 1
        State 1 --p10--> State 0
    
    Memory complexity: Cμ ≈ H(π) where π is stationary distribution
    
    Args:
        transition_prob_01: Probability of 0→1 transition
        transition_prob_10: Probability of 1→0 transition
        
    Returns:
        Generator function that produces Markov sequences
    """
    def generate_sequence(length, *, seed=0):
        rng = create_random_number_generator(seed)
        sequence = np.empty(length, int)
        
        # Random initial state
        sequence[0] = rng.integers(2)
        
        for time_step in range(1, length):
            previous_symbol = sequence[time_step - 1]
            
            if previous_symbol == 0 and rng.random() < transition_prob_01:
                sequence[time_step] = 1
            elif previous_symbol == 1 and rng.random() < transition_prob_10:
                sequence[time_step] = 0
            else:
                sequence[time_step] = previous_symbol
                
        return sequence
    
    return generate_sequence


def create_golden_mean_process(emission_prob_01=0.5):
    """
    Factory for Golden Mean process (sofic).
    
    Constraint: Forbids consecutive ones "11"
    Memory complexity: Cμ = 1 bit
    
    Args:
        emission_prob_01: Probability of emitting '1' after '0'
        
    Returns:
        Generator function that produces Golden Mean sequences
    """
    def generate_sequence(length, *, seed=0):
        rng = create_random_number_generator(seed)
        sequence = np.empty(length, int)
        
        # Random initial symbol
        sequence[0] = rng.integers(2)
        
        for time_step in range(1, length):
            previous_symbol = sequence[time_step - 1]
            
            if previous_symbol == 1:
                # Must emit 0 after 1 (no consecutive 1s allowed)
                sequence[time_step] = 0
            else:
                # After 0, can emit 1 with given probability
                sequence[time_step] = 1 if rng.random() < emission_prob_01 else 0
                
        return sequence
    
    return generate_sequence


def create_even_process(transition_prob_10=0.5):
    """
    Factory for Even process (sofic).
    
    Constraint: Blocks of 0s between successive 1s must have even length
    Memory complexity: Cμ = 1 bit (minimal epsilon-machine has two states)
    
    Args:
        transition_prob_10: Probability of transitioning from 1 to 0
        
    Returns:
        Generator function that produces Even sequences
    """
    def generate_sequence(length, *, seed=0):
        rng = create_random_number_generator(seed)
        sequence = np.empty(length, int)
        
        # Track whether we're in an even-length block state
        in_even_block = True
        
        for time_step in range(length):
            if in_even_block:
                if rng.random() < transition_prob_10:
                    sequence[time_step] = 1
                else:
                    sequence[time_step] = 0
                    in_even_block = False  # Start odd block
            else:
                # Must emit 0 and return to even block state
                sequence[time_step] = 0
                in_even_block = True
                
        return sequence
    
    return generate_sequence


def create_thue_morse_process():
    """
    Factory for deterministic Thue-Morse sequence.
    
    Construction: Iterative morphism 0→01, 1→10
    Memory complexity: Cμ grows logarithmically
    
    Returns:
        Generator function that produces Thue-Morse sequences
    """
    def generate_sequence(length, *, seed=0):
        # Seed parameter ignored (deterministic process)
        sequence = [0]
        
        # Iteratively apply morphism until we have enough symbols
        while len(sequence) < length:
            # Apply morphism: flip each bit and append
            morphed_sequence = [1 - symbol for symbol in sequence]
            sequence.extend(morphed_sequence)
            
        return np.array(sequence[:length], int)
    
    return generate_sequence


# Factory dictionary for easy access to all processes
PROCESS_GENERATORS = {
    # Zero/low complexity processes
    "IID(p=0.5)": create_iid_process(0.5),
    "IID(p=0.25)": create_iid_process(0.25),
    "Two-state Markov": create_two_state_markov_process(0.3, 0.7),
    
    # Sofic / finite-state epsilon-machines
    "Golden-Mean": create_golden_mean_process(),
    "Even": create_even_process(),
    
    # High / unbounded complexity processes
    "Thue-Morse": create_thue_morse_process(),
}


def validate_process(process_name: str, sequence: np.ndarray) -> bool:
    """
    Basic validation that a process behaves as expected.
    """
    if process_name == "Golden-Mean":
        # Check for no consecutive ones
        for i in range(len(sequence) - 1):
            if sequence[i] == 1 and sequence[i+1] == 1:
                return False
    
    return True


if __name__ == "__main__":
    # Demonstration: generate and display first 40 symbols for each process
    np.set_printoptions(threshold=40, linewidth=120)
    
    for process_name, generator in PROCESS_GENERATORS.items():
        sequence = generator(40, seed=1)
        is_valid = validate_process(process_name, sequence)
        print(f"{process_name:>20}: {sequence} (valid: {is_valid})")