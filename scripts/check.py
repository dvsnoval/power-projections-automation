#!/usr/bin/env python
"""Development script for running quality checks."""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report the result."""
    print(f"\n🔍 {description}...")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} passed")
        if result.stdout.strip():
            print(result.stdout)
    else:
        print(f"❌ {description} failed")
        print(result.stdout)
        print(result.stderr)
        return False
    return True

def main():
    """Run all quality checks."""
    project_root = Path(__file__).parent
    code_dir = project_root / "code"
    tests_dir = project_root / "tests"
    
    checks = [
        (["black", "--check", str(code_dir), str(tests_dir)], "Code formatting (black)"),
        (["isort", "--check-only", str(code_dir), str(tests_dir)], "Import sorting (isort)"),
        (["flake8", str(code_dir), str(tests_dir)], "Code linting (flake8)"),
        (["mypy", str(code_dir)], "Type checking (mypy)"),
        (["pytest", "--cov=code", "--cov-report=term-missing"], "Tests with coverage"),
    ]
    
    print("🚀 Running quality checks...")
    failed_checks = []
    
    for cmd, description in checks:
        if not run_command(cmd, description):
            failed_checks.append(description)
    
    if failed_checks:
        print(f"\n❌ {len(failed_checks)} check(s) failed:")
        for check in failed_checks:
            print(f"  - {check}")
        sys.exit(1)
    else:
        print(f"\n🎉 All {len(checks)} quality checks passed!")

if __name__ == "__main__":
    main()