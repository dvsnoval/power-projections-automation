#!/usr/bin/env python
"""Development script for formatting code."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report the result."""
    print(f"\n🔧 {description}...")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ {description} completed")
        if result.stdout.strip():
            print(result.stdout)
    else:
        print(f"❌ {description} failed")
        print(result.stdout)
        print(result.stderr)
        return False
    return True


def main():
    """Format all code."""
    project_root = Path(__file__).parent.parent
    code_dir = project_root / "utils"
    tests_dir = project_root / "tests"
    scripts_dir = project_root / "scripts"

    formatters = [
        (
            ["black", str(code_dir), str(tests_dir), str(scripts_dir)],
            "Code formatting (black)",
        ),
        (
            ["isort", str(code_dir), str(tests_dir), str(scripts_dir)],
            "Import sorting (isort)",
        ),
    ]

    print("🎨 Formatting code...")
    failed_formatters = []

    for cmd, description in formatters:
        if not run_command(cmd, description):
            failed_formatters.append(description)

    if failed_formatters:
        print(f"\n❌ {len(failed_formatters)} formatter(s) failed:")
        for formatter in failed_formatters:
            print(f"  - {formatter}")
        sys.exit(1)
    else:
        print("\n🎉 Code formatting completed successfully!")


if __name__ == "__main__":
    main()
