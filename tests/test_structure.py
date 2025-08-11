#!/usr/bin/env python3
"""
Test script to verify simplified modular structure works.
Tests basic imports and functionality without requiring external dependencies.
"""

import sys
import os

# Add src to path (adjust for tests/ directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_basic_imports():
    """Test that all modules can be imported."""
    print("Testing basic module imports...")
    
    # Core building blocks
    try:
        import interfaces
        print("✓ interfaces imported successfully")
    except ImportError as e:
        print(f"✗ interfaces import failed: {e}")
        return False
    
    try:
        import implementations
        print("✓ implementations imported successfully")
    except ImportError as e:
        print(f"✗ implementations import failed: {e}")
        return False
    
    # Process generators
    try:
        import processes
        print("✓ processes imported successfully")
    except ImportError as e:
        print(f"✗ processes import failed: {e}")
        return False
    
    # Core algorithms
    try:
        import empirical_analysis
        print("✓ empirical_analysis imported successfully")
    except ImportError as e:
        print(f"✗ empirical_analysis import failed: {e}")
        return False
    
    try:
        import information_theory
        print("✓ information_theory imported successfully")
    except ImportError as e:
        print(f"✗ information_theory import failed: {e}")
        return False
    
    try:
        import expectation_maximization
        print("✓ expectation_maximization imported successfully")
    except ImportError as e:
        print(f"✗ expectation_maximization import failed: {e}")
        return False
    
    try:
        import state_splitting
        print("✓ state_splitting imported successfully")
    except ImportError as e:
        print(f"✗ state_splitting import failed: {e}")
        return False
    
    # High-level analysis (now in scripts/)
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
        import batch
        print("✓ batch (from scripts/) imported successfully")
    except ImportError as e:
        print(f"✗ batch import failed: {e}")
        return False
    
    try:
        import online
        print("✓ online (from scripts/) imported successfully")
    except ImportError as e:
        print(f"✗ online import failed: {e}")
        return False
    
    # Visualization
    try:
        import visualization
        print("✓ visualization imported successfully")
    except ImportError as e:
        print(f"✗ visualization import failed: {e}")
        return False
    
    # Demo (now in scripts/)
    try:
        import demo
        print("✓ demo (from scripts/) imported successfully")
    except ImportError as e:
        print(f"✗ demo import failed: {e}")
        return False

    return True

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\nTesting basic functionality...")
    
    try:
        from processes import PROCESS_GENERATORS, create_iid_process
        
        # Test IID process generation
        iid_generator = create_iid_process(0.5)
        sequence = iid_generator(100, seed=42)
        print(f"✓ Generated IID sequence: length {len(sequence)}")
        
        # Test process factory
        print(f"✓ Available processes: {len(PROCESS_GENERATORS)}")
        for name in PROCESS_GENERATORS.keys():
            print(f"  - {name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_interfaces():
    """Test that interfaces are properly defined."""
    print("\nTesting interfaces...")
    
    try:
        from interfaces import (
            ProcessGenerator, ProbabilityAnalyzer, InformationCalculator,
            EMOptimizer, StateSplitter, InformationBottleneckAnalyzer,
            OnlineProcessor, AnalysisFactory, Visualizer
        )
        
        print("✓ All interfaces imported successfully")
        print(f"✓ ProcessGenerator protocol defined")
        print(f"✓ Analysis interfaces defined")
        print(f"✓ Factory interface defined")
        print(f"✓ Visualizer interface defined")
        
        return True
        
    except Exception as e:
        print(f"✗ Interface test failed: {e}")
        return False

def test_factory_pattern():
    """Test that factory pattern works."""
    print("\nTesting factory pattern...")
    
    try:
        from implementations import StandardAnalysisFactory
        
        factory = StandardAnalysisFactory()
        print("✓ Factory created successfully")
        
        # Test creating components
        prob_analyzer = factory.create_probability_analyzer()
        info_calc = factory.create_information_calculator()
        em_optimizer = factory.create_em_optimizer(info_calc)
        splitter = factory.create_state_splitter(info_calc, em_optimizer)
        
        print("✓ All components created via factory")
        
        return True
        
    except Exception as e:
        print(f"✗ Factory pattern test failed: {e}")
        return False

def test_composition():
    """Test that modules compose correctly."""
    print("\nTesting composition...")
    
    try:
        from implementations import StandardAnalysisFactory
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
        from batch import BatchAnalyzer
        
        factory = StandardAnalysisFactory()
        
        # Create analyzer with dependency injection
        analyzer = BatchAnalyzer(
            factory.create_probability_analyzer(),
            factory.create_information_calculator(),
            factory.create_em_optimizer(factory.create_information_calculator()),
            factory.create_state_splitter(
                factory.create_information_calculator(),
                factory.create_em_optimizer(factory.create_information_calculator())
            )
        )
        
        print("✓ Batch analyzer composed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Composition test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("SIMPLIFIED STRUCTURE VALIDATION")
    print("="*60)
    
    tests = [
        test_basic_imports,
        test_basic_functionality,
        test_interfaces,
        test_factory_pattern,
        test_composition
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"FAILED: {test.__name__}")
        except Exception as e:
            print(f"ERROR in {test.__name__}: {e}")
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Simplified structure is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)