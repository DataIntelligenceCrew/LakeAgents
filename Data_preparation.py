#!/usr/bin/env python3
"""
Simple data preparation pipeline
Runs selected data processing scripts in sequence

Usage:
    python Data_preparation.py all                    # Run all scripts
    python Data_preparation.py llm subtables join compare  # Run specific scripts
    python Data_preparation.py subtables join compare     # Run specific scripts (skip llm)
"""
from compare_rejoined_original_tables import get_successful_rejoined_tables
import subprocess
import sys
import os
import json  

def get_verified_tables(datasets_dir="datasets"):
    """get rejoined table names
    
    Args:
        datasets_dir: Directory containing the datasets (default: "datasets")
    """
    successful_tables = get_successful_rejoined_tables(datasets_dir)
    return successful_tables

def save_verified_tables_to_file(verified_tables):
    """Save verified tables to a JSON file for later use"""
    output_file = "verified_tables.json"
    
    # Load LLM configuration if available
    llm_config = load_llm_config()
    
    # Create output with metadata
    output_data = {
        "verified_tables": verified_tables,
        "total_count": len(verified_tables),
        "llm_config": llm_config if llm_config else "Not available (LLM step was skipped or config file missing)"
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Verified tables saved to: {output_file}")
    if llm_config:
        print(f"  LLM Provider: {llm_config.get('llm_provider', 'N/A')}")
        print(f"  LLM Model: {llm_config.get('llm_model', 'N/A')}")
    return output_file

def load_verified_tables_from_file():
    """Load verified tables from saved JSON file"""
    output_file = "verified_tables.json"
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            verified_tables = data.get('verified_tables', [])
        else:
            verified_tables = data
        print(f"Loaded {len(verified_tables)} verified tables from {output_file}")
        return verified_tables
    else:
        print(f"No verified tables file found: {output_file}")
        return []


def save_llm_config(provider, model, max_datasets):
    """Save LLM configuration to file"""
    config = {
        "llm_provider": provider,
        "llm_model": model if model else f"default ({provider})",
        "max_datasets": max_datasets
    }
    
    config_file = "llm_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"LLM configuration saved to: {config_file}")
    return config

def load_llm_config():
    """Load LLM configuration from file"""
    config_file = "llm_config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return None

def run_llm_step(provider="openai", model=None, max_datasets=50, datasets_dir="datasets"):
    """Run LLM prompt step
    
    Args:
        provider: "openai" or "gemini"
        model: Model name (optional)
        max_datasets: Maximum number of datasets to process
        datasets_dir: Directory containing the datasets (default: "datasets")
    """
    print("=== Running LLM Step ===")
    print(f"Provider: {provider.upper()}")
    if model:
        print(f"Model: {model}")
    print(f"Max datasets: {max_datasets}")
    print(f"Datasets directory: {datasets_dir}")
    
    # Save LLM configuration
    save_llm_config(provider, model, max_datasets)
    
    cmd = ["python", "test_llm_prompt.py", f"{datasets_dir}/", "--max", str(max_datasets), 
           "--provider", provider]
    
    if model:
        cmd.extend(["--model", model])
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def run_subtables_step(datasets_dir="datasets"):
    """Run subtables step
    
    Args:
        datasets_dir: Directory containing the datasets (default: "datasets")
    """
    print("=== Running Subtables Step ===")
    print(f"Datasets directory: {datasets_dir}")
    cmd = ["python", "subtables.py", "--datasets-dir", datasets_dir]
    result = subprocess.run(cmd)
    return result.returncode == 0

def run_join_step(datasets_dir="datasets"):
    """Run join step
    
    Args:
        datasets_dir: Directory containing the datasets (default: "datasets")
    """
    print("=== Running Join Step ===")
    print(f"Datasets directory: {datasets_dir}")
    cmd = ["python", "join.py", "--datasets-dir", datasets_dir]
    result = subprocess.run(cmd)
    return result.returncode == 0

def run_compare_step(datasets_dir="datasets"):
    """Run compare step and save verified tables
    
    Args:
        datasets_dir: Directory containing the datasets (default: "datasets")
    """
    print("=== Running Compare Step ===")
    print(f"Datasets directory: {datasets_dir}")

    # Import the function and call it with datasets_dir
    from compare_rejoined_original_tables import main as compare_main_func
    verified_tables = compare_main_func(datasets_dir)
    
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

def run_data_preparation_steps(script_names, provider="openai", model=None, max_datasets=50, datasets_dir="datasets"):
    """Run selected data preparation steps
    
    Args:
        script_names: List of script names to run
        provider: LLM provider ("openai" or "gemini")
        model: Model name (optional)
        max_datasets: Maximum number of datasets to process
        datasets_dir: Directory containing the datasets (default: "datasets")
    """
    
    # Define available steps
    available_steps = {
        "llm": lambda: run_llm_step(provider=provider, model=model, max_datasets=max_datasets, datasets_dir=datasets_dir),
        "subtables": lambda: run_subtables_step(datasets_dir=datasets_dir),
        "join": lambda: run_join_step(datasets_dir=datasets_dir),
        "compare": lambda: run_compare_step(datasets_dir=datasets_dir)
    }
    
    # Validate script names
    invalid_scripts = [name for name in script_names if name not in available_steps]
    if invalid_scripts:
        print(f"Error: Invalid script names: {invalid_scripts}")
        print(f"Available scripts: {list(available_steps.keys())}")
        return False
    
    # Print what we're going to run
    print(f"=== RUNNING SCRIPTS: {', '.join(script_names)} ===")
    print(f"Datasets directory: {datasets_dir}")
    if "llm" in script_names:
        print(f"LLM Configuration: Provider={provider.upper()}, Model={model or 'default'}, Max={max_datasets}")
    else:
        # If LLM step is skipped, show existing LLM config if available
        llm_config = load_llm_config()
        if llm_config:
            print(f"Note: Using existing LLM results from previous run:")
            print(f"  Provider: {llm_config.get('llm_provider', 'N/A')}")
            print(f"  Model: {llm_config.get('llm_model', 'N/A')}")
        else:
            print("Note: LLM step skipped, no previous LLM config found")
    
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
        print("  python Data_preparation.py all [options]")
        print("  python Data_preparation.py llm subtables join compare [options]")
        print("  python Data_preparation.py subtables join compare [options]")
        print("\nAvailable scripts: llm, subtables, join, compare")
        print("\nOptions:")
        print("  --provider openai|gemini    LLM provider (default: openai)")
        print("  --model MODEL_NAME          Model name (optional)")
        print("  --max N                     Maximum datasets to process (default: 50)")
        print("  --datasets-dir DIR          Datasets directory (default: datasets)")
        print("\nExamples:")
        print("  python Data_preparation.py all --provider openai")
        print("  python Data_preparation.py all --provider gemini --model gemini-2.0-flash-exp")
        print("  python Data_preparation.py llm subtables --provider openai --max 100")
        print("  python Data_preparation.py subtables join compare  # Skip LLM step")
        print("  python Data_preparation.py all --datasets-dir datasets2  # Use datasets2 folder")
        sys.exit(1)
    
    # Parse arguments
    args = sys.argv[1:]
    script_names = []
    provider = "openai"
    model = None
    max_datasets = 50
    # Try to find the correct datasets directory
    datasets_dir = "datasets"
    # Check if default directory exists, if not try alternatives
    if not os.path.exists(datasets_dir):
        if os.path.exists("datasets_omnimatch2"):
            datasets_dir = "datasets_omnimatch2"
            print(f"Note: 'datasets' directory not found, using '{datasets_dir}' instead")
        elif os.path.exists("datasets2"):
            datasets_dir = "datasets2"
            print(f"Note: 'datasets' directory not found, using '{datasets_dir}' instead")
    
    # Extract script names and options
    i = 0
    while i < len(args):
        arg = args[i]
        
        if arg == "--provider" and i + 1 < len(args):
            provider = args[i + 1]
            i += 2
        elif arg == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif arg == "--max" and i + 1 < len(args):
            max_datasets = int(args[i + 1])
            i += 2
        elif arg == "--datasets-dir" and i + 1 < len(args):
            datasets_dir = args[i + 1]
            i += 2
        elif arg.startswith("--"):
            print(f"Unknown option: {arg}")
            sys.exit(1)
        else:
            script_names.append(arg)
            i += 1
    
    # Handle 'all' parameter
    if "all" in script_names:
        script_names = ["llm", "subtables", "join", "compare"]
    
    # Validate we have at least one script
    if not script_names:
        print("Error: No scripts specified")
        sys.exit(1)
    
    # Run the steps
    success = run_data_preparation_steps(script_names, provider=provider, model=model, max_datasets=max_datasets, datasets_dir=datasets_dir)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()