#!/usr/bin/env python3
"""
Test runner script for Match Video Detection project.
This script provides easy access to run tests with different configurations.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(command, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run tests for Match Video Detection project")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "all"], 
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--fast", 
        action="store_true",
        help="Run only fast tests (skip slow markers)"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--install-deps", 
        action="store_true",
        help="Install test dependencies first"
    )
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🧪 Match Video Detection Test Runner")
    print(f"Project root: {project_root}")
    
    # Install test dependencies if requested
    if args.install_deps:
        print("\n📦 Installing test dependencies...")
        if not run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                          "Installing dependencies"):
            return 1
    
    # Build pytest command
    pytest_cmd = [sys.executable, "-m", "pytest"]
    
    # Add coverage if requested
    if args.coverage:
        pytest_cmd.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html:tests/coverage"])
    
    # Add verbose flag
    if args.verbose:
        pytest_cmd.append("-v")
    
    # Add parallel execution
    if args.parallel:
        pytest_cmd.extend(["-n", "auto"])
    
    # Add test type filters
    if args.type == "unit":
        pytest_cmd.extend(["-m", "unit"])
    elif args.type == "integration":
        pytest_cmd.extend(["-m", "integration"])
    
    # Skip slow tests if fast mode
    if args.fast:
        pytest_cmd.extend(["-m", "not slow"])
    
    # Run tests
    test_description = f"{args.type.title()} tests"
    if args.coverage:
        test_description += " with coverage"
    if args.fast:
        test_description += " (fast mode)"
    if args.parallel:
        test_description += " (parallel)"
    
    success = run_command(pytest_cmd, test_description)
    
    if success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    import os
    sys.exit(main()) 