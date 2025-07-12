#!/usr/bin/env python3
"""
Code formatting and linting script for Match Video Detection project.
This script provides easy access to format and check code quality.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(command, check=check, capture_output=False)
        if result.returncode == 0:
            print(f"\n✅ {description} completed successfully!")
            return True
        else:
            print(f"\n⚠️  {description} completed with warnings/errors")
            return False
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    """Main code formatting function."""
    parser = argparse.ArgumentParser(description="Format and lint Match Video Detection code")
    parser.add_argument(
        "--format", 
        action="store_true",
        help="Format code with Black and isort"
    )
    parser.add_argument(
        "--lint", 
        action="store_true",
        help="Run linting checks (flake8, ruff, mypy)"
    )
    parser.add_argument(
        "--check", 
        action="store_true",
        help="Check code without making changes"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Run all formatting and linting"
    )
    parser.add_argument(
        "--install-hooks", 
        action="store_true",
        help="Install pre-commit hooks"
    )
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🎨 Match Video Detection Code Quality Tools")
    print(f"Project root: {project_root}")
    
    # Install pre-commit hooks if requested
    if args.install_hooks:
        print("\n📦 Installing pre-commit hooks...")
        if not run_command([sys.executable, "-m", "pre_commit", "install"], 
                          "Installing pre-commit hooks"):
            return 1
    
    # Determine what to run
    run_format = args.format or args.all
    run_lint = args.lint or args.all
    
    if not (run_format or run_lint or args.install_hooks):
        print("\n❌ No action specified. Use --help for options.")
        return 1
    
    success = True
    
    # Code formatting
    if run_format:
        print("\n🎨 Code Formatting")
        
        # Black formatting
        black_cmd = [sys.executable, "-m", "black"]
        if args.check:
            black_cmd.append("--check")
        black_cmd.extend(["src", "tests", "scripts"])
        
        if not run_command(black_cmd, "Black code formatting", check=False):
            success = False
        
        # isort import sorting
        isort_cmd = [sys.executable, "-m", "isort"]
        if args.check:
            isort_cmd.append("--check-only")
        isort_cmd.extend(["src", "tests", "scripts"])
        
        if not run_command(isort_cmd, "isort import sorting", check=False):
            success = False
    
    # Code linting
    if run_lint:
        print("\n🔍 Code Linting")
        
        # Ruff linting
        ruff_cmd = [sys.executable, "-m", "ruff", "check"]
        if not args.check:
            ruff_cmd.append("--fix")
        ruff_cmd.extend(["src", "tests", "scripts"])
        
        if not run_command(ruff_cmd, "Ruff linting", check=False):
            success = False
        
        # Flake8 linting
        flake8_cmd = [sys.executable, "-m", "flake8", "src", "tests", "scripts"]
        if not run_command(flake8_cmd, "Flake8 linting", check=False):
            success = False
        
        # MyPy type checking
        mypy_cmd = [sys.executable, "-m", "mypy", "src"]
        if not run_command(mypy_cmd, "MyPy type checking", check=False):
            success = False
    
    # Summary
    if success:
        print("\n🎉 All code quality checks passed!")
        return 0
    else:
        print("\n💥 Some code quality checks failed!")
        print("\n💡 Tips:")
        print("  - Run 'python scripts/format_code.py --format' to fix formatting")
        print("  - Run 'python scripts/format_code.py --lint' to see linting issues")
        print("  - Run 'python scripts/format_code.py --all' to format and lint")
        return 1


if __name__ == "__main__":
    import os
    sys.exit(main()) 