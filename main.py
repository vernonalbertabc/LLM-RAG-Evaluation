import os
import subprocess
import sys
from pathlib import Path

def main():
    """
    Main function to execute RAG evaluation experiments.
    """
    print("Starting RAG Evaluation Experiments...")
    print("=" * 50)
    
    # Get the current directory
    current_dir = Path(__file__).parent
    experiments_dir = current_dir / "src" / "llm_evaluation"
    
    # Check if the experiments directory exists
    if not experiments_dir.exists():
        print(f"Error: Experiments directory not found at {experiments_dir}")
        sys.exit(1)
    
    # Change to the experiments directory
    os.chdir(experiments_dir)
    print(f"Changed to directory: {experiments_dir}")
    
    # Check command line arguments for experiment type
    if len(sys.argv) > 1:
        experiment_type = sys.argv[1].lower()
    else:
        # Default to chunk_size experiments if no argument provided
        experiment_type = "chunk_size"
        print("No experiment type specified, defaulting to 'chunk_size'")
    
    # Validate experiment type
    valid_types = ["chunk_size", "contextualization"]
    if experiment_type not in valid_types:
        print(f"Error: Invalid experiment type '{experiment_type}'")
        print(f"Valid types: {', '.join(valid_types)}")
        sys.exit(1)
    
    # Execute the experiments
    try:
        print(f"\nExecuting {experiment_type} experiments...")
        print(f"Command: python run_experiments.py {experiment_type}")
        print("-" * 50)
        
        # Run the experiments
        result = subprocess.run(
            ["python", "run_experiments.py", experiment_type],
            capture_output=False,  # Show output in real-time
            text=True,
            cwd=experiments_dir
        )
        
        if result.returncode == 0:
            print("\n" + "=" * 50)
            print(f"{experiment_type.capitalize()} experiments completed successfully!")
            print("Check MLflow UI at http://localhost:5000 to view results")
        else:
            print(f"\nExperiments failed with return code: {result.returncode}")
            sys.exit(1)
            
    except FileNotFoundError:
        print("Error: run_experiments.py not found in the experiments directory")
        sys.exit(1)
    except Exception as e:
        print(f"Error executing experiments: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
