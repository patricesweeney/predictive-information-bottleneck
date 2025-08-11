"""
State Splitting Validation Test Runner
=====================================

Comprehensive test runner for validating state splitting correctness.
Runs all validation tests and provides summary report.
"""

import sys
import os
import pytest
import time
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class ValidationTestRunner:
    """Coordinated test runner for state splitting validation."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.total_time = None
    
    def run_all_tests(self, verbose=True):
        """Run all validation test suites."""
        self.start_time = time.time()
        
        test_suites = [
            ("Ground Truth Validation", "unit/test_state_splitting_correctness.py"),
            ("Epsilon Machine Validation", "unit/test_epsilon_machine_validation.py"),
            ("Method Consistency", "integration/test_method_consistency.py"),
            ("Robustness Tests", "integration/test_robustness.py")
        ]
        
        print("=" * 60)
        print("STATE SPLITTING VALIDATION TEST SUITE")
        print("=" * 60)
        print()
        
        for suite_name, test_file in test_suites:
            print(f"Running {suite_name}...")
            print("-" * 40)
            
            success, details = self._run_test_suite(test_file, verbose)
            self.results[suite_name] = {
                'success': success,
                'details': details
            }
            
            if success:
                print(f"✅ {suite_name}: PASSED")
            else:
                print(f"❌ {suite_name}: FAILED")
                if not verbose:
                    print(f"   {details}")
            print()
        
        self.total_time = time.time() - self.start_time
        self._print_summary()
    
    def _run_test_suite(self, test_file, verbose):
        """Run a single test suite."""
        test_path = os.path.join(os.path.dirname(__file__), test_file)
        
        if not os.path.exists(test_path):
            return False, f"Test file not found: {test_path}"
        
        # Capture pytest output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        try:
            if verbose:
                # Run with live output
                result = pytest.main([test_path, "-v", "--tb=short"])
            else:
                # Capture output
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    result = pytest.main([test_path, "-v", "--tb=short", "-q"])
            
            success = (result == 0)
            
            if not success and not verbose:
                # Return captured error output
                error_output = stderr_capture.getvalue()
                if not error_output:
                    error_output = stdout_capture.getvalue()
                return False, error_output[:500]  # Truncate long errors
            
            return success, "All tests passed" if success else "Some tests failed"
            
        except Exception as e:
            return False, f"Error running tests: {str(e)}"
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def _print_summary(self):
        """Print test summary."""
        print("=" * 60)
        print("VALIDATION TEST SUMMARY")
        print("=" * 60)
        
        total_suites = len(self.results)
        passed_suites = sum(1 for r in self.results.values() if r['success'])
        failed_suites = total_suites - passed_suites
        
        print(f"Total test suites: {total_suites}")
        print(f"Passed: {passed_suites}")
        print(f"Failed: {failed_suites}")
        print(f"Total time: {self.total_time:.1f}s")
        print()
        
        if failed_suites == 0:
            print("🎉 ALL VALIDATION TESTS PASSED!")
            print("✅ State splitting correctness validated")
        else:
            print("⚠️  SOME VALIDATION TESTS FAILED")
            print("❌ State splitting needs attention")
            print()
            print("Failed suites:")
            for suite_name, result in self.results.items():
                if not result['success']:
                    print(f"  - {suite_name}: {result['details']}")
        
        print("=" * 60)
    
    def run_quick_validation(self):
        """Run a quick subset of tests for fast validation."""
        print("Running Quick Validation Tests...")
        print("=" * 40)
        
        # Run key tests from each suite
        quick_tests = [
            "unit/test_state_splitting_correctness.py::TestGroundTruthValidation::test_golden_mean_discovers_two_states",
            "unit/test_epsilon_machine_validation.py::TestEpsilonMachineStructure::test_golden_mean_causal_states",
            "integration/test_method_consistency.py::TestOnlineVsBatchConsistency::test_golden_mean_online_vs_batch_states"
        ]
        
        all_passed = True
        for test in quick_tests:
            test_path = os.path.join(os.path.dirname(__file__), test)
            result = pytest.main([test_path, "-v", "--tb=short", "-q"])
            
            if result == 0:
                print(f"✅ {test.split('::')[-1]}")
            else:
                print(f"❌ {test.split('::')[-1]}")
                all_passed = False
        
        print()
        if all_passed:
            print("🎉 Quick validation PASSED!")
        else:
            print("❌ Quick validation FAILED - run full tests")
    
    def generate_validation_report(self, output_file="validation_report.txt"):
        """Generate a detailed validation report."""
        if not self.results:
            print("No test results available. Run tests first.")
            return
        
        report_path = os.path.join(os.path.dirname(__file__), output_file)
        
        with open(report_path, 'w') as f:
            f.write("STATE SPLITTING VALIDATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total test time: {self.total_time:.1f}s\n\n")
            
            f.write("TEST RESULTS:\n")
            f.write("-" * 20 + "\n")
            
            for suite_name, result in self.results.items():
                status = "PASSED" if result['success'] else "FAILED"
                f.write(f"{suite_name:30}: {status}\n")
                if not result['success']:
                    f.write(f"  Details: {result['details'][:200]}\n")
            
            f.write("\nVALIDATION CRITERIA:\n")
            f.write("-" * 20 + "\n")
            f.write("✅ Ground Truth: Known processes discover correct state counts\n")
            f.write("✅ Epsilon Machines: States match causal structure\n")
            f.write("✅ Method Consistency: Online vs batch agreement\n")
            f.write("✅ Robustness: Stable under noise and parameter changes\n")
            
            # Overall assessment
            passed_count = sum(1 for r in self.results.values() if r['success'])
            total_count = len(self.results)
            
            f.write(f"\nOVERALL ASSESSMENT:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Test suites passed: {passed_count}/{total_count}\n")
            
            if passed_count == total_count:
                f.write("✅ STATE SPLITTING VALIDATION: COMPLETE\n")
                f.write("   State discovery algorithms are working correctly.\n")
            else:
                f.write("❌ STATE SPLITTING VALIDATION: INCOMPLETE\n")
                f.write("   Some validation criteria not met.\n")
        
        print(f"Validation report saved to: {report_path}")


def main():
    """Main test runner entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run state splitting validation tests")
    parser.add_argument("--quick", action="store_true", help="Run quick validation only")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--report", action="store_true", help="Generate validation report")
    
    args = parser.parse_args()
    
    runner = ValidationTestRunner()
    
    if args.quick:
        runner.run_quick_validation()
    else:
        runner.run_all_tests(verbose=not args.quiet)
        
        if args.report:
            runner.generate_validation_report()


if __name__ == "__main__":
    main()