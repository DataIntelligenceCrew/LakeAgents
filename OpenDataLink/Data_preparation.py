#!/usr/bin/env python3
"""
Simple data preparation pipeline
Runs selected data processing scripts in sequence

Usage:
    python Data_preparation.py all                    # Run all scripts
    python Data_preparation.py llm subtables join compare  # Run specific scripts
    python Data_preparation.py subtables join compare     # Run specific scripts (skip llm)
"""
from compare_rejoined_original_tables import get_successful_rejoined_tables, main as compare_main
import subprocess
import sys
import os
import json  

def get_verified_tables():
    """get rejoined table names"""
    successful_tables = get_successful_rejoined_tables()
    return successful_tables

def save_verified_tables_to_file(verified_tables):
    """Save verified tables to a JSON file for later use"""
    output_file = "verified_tables.json"
    with open(output_file, 'w') as f:
        json.dump(verified_tables, f, indent=2)
    print(f"Verified tables saved to: {output_file}")
    return output_file

def load_verified_tables_from_file():
    """Load verified tables from saved JSON file"""
    output_file = "verified_tables.json"
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            verified_tables = json.load(f)
        print(f"Loaded {len(verified_tables)} verified tables from {output_file}")
        return verified_tables
    else:
        print(f"No verified tables file found: {output_file}")
        return []


def run_llm_step():
    """Run LLM prompt step"""
    print("=== Running LLM Step ===")
    cmd = ["python", "test_llm_prompt.py", "datasets/", "--max", "50"]
    result = subprocess.run(cmd)
    return result.returncode == 0

def run_subtables_step():
    """Run subtables step"""
    print("=== Running Subtables Step ===")
    cmd = ["python", "subtables.py"]
    result = subprocess.run(cmd)
    return result.returncode == 0

def run_join_step():
    """Run join step"""
    print("=== Running Join Step ===")
    cmd = ["python", "join.py"]
    result = subprocess.run(cmd)
    return result.returncode == 0

def run_compare_step():
    """Run compare step and save verified tables"""
    print("=== Running Compare Step ===")

    verified_tables = compare_main()
    
    if verified_tables is not None:
        # Save verified tables
        print("=== Saving Verified Tables ===")
        if verified_tables:
            save_verified_tables_to_file(verified_tables)
        else:
            print("No verified tables found")
        return True
    else:
        return False

def run_data_preparation_steps(script_names):
    """Run selected data preparation steps"""
    
    # Define available steps
    available_steps = {
        "llm": run_llm_step,
        "subtables": run_subtables_step,
        "join": run_join_step,
        "compare": run_compare_step
    }
    
    # Validate script names
    invalid_scripts = [name for name in script_names if name not in available_steps]
    if invalid_scripts:
        print(f"Error: Invalid script names: {invalid_scripts}")
        print(f"Available scripts: {list(available_steps.keys())}")
        return False
    
    # Print what we're going to run
    print(f"=== RUNNING SCRIPTS: {', '.join(script_names)} ===")
    
    # Run selected scripts
    for i, script_name in enumerate(script_names, 1):
        print(f"\n=== STEP {i}: Running {script_name} ===")
        
        success = available_steps[script_name]()
        
        if not success:
            print(f"Error in step {i} ({script_name}). Stopping.")
            return False
        
        print(f"Step {i} ({script_name}) completed successfully!")
    
    print(f"\n=== ALL SELECTED STEPS COMPLETED ===")
    print(f"Successfully ran: {', '.join(script_names)}")
    return True

def main():
    """Run selected data preparation scripts"""
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python Data_preparation.py all")
        print("  python Data_preparation.py llm subtables join compare")
        print("  python Data_preparation.py subtables join compare")
        print("\nAvailable scripts: llm, subtables, join, compare")
        sys.exit(1)
    
    # Get script names from command line
    script_names = sys.argv[1:]
    
    # Handle 'all' parameter
    if "all" in script_names:
        script_names = ["llm", "subtables", "join", "compare"]
    
    # Run the steps
    success = run_data_preparation_steps(script_names)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()