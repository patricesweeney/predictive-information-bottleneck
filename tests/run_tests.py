#!/usr/bin/env python3
"""
Test Runner for Predictive Information Bottleneck
================================================

Runs all tests in the correct order and provides a summary.
"""

import sys
import os
import subprocess
from pathlib import Path

def run_test_file(test_file):
    """Run a single test file and return success status."""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=False, 
                              text=True, 
                              cwd=os.path.dirname(test_file))
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {test_file}: {e}")
        return False

def main():
    """Run all tests."""
    print("Predictive Information Bottleneck - Test Suite")
    print("=" * 60)
    
    # Get test directory
    test_dir = Path(__file__).parent
    
    # Define test order (structure before composition)
    test_files = [
        test_dir / "test_structure.py",
        test_dir / "test_composition.py"
    ]
    
    # Run tests
    results = []
    for test_file in test_files:
        if test_file.exists():
            success = run_test_file(str(test_file))
            results.append((test_file.name, success))
        else:
            print(f"Warning: {test_file} not found")
            results.append((test_file.name, False))
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:25} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} test files passed")
    
    if passed == total:
        print("🎉 All tests passed! Architecture is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)