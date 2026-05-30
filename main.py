"""
main.py
Mia Shuteva-89231346

Usage:
    python main.py sudoku.csv
    python main.py sudoku.csv --samples 2000 --epochs 5

This script:
  1. Loads the dataset
  2. Trains the ANN
  3. Evaluates CSP, ANN, and Hybrid solvers
  4. Generates all statistics and figures
"""

import argparse
from sudoku_solver import run_pipeline
from statistics_analysis import analyse


def main():
    #Allows the user to input the path to the CSV file, number of samples, and number of
    #ANN training epochs via command line argument
    parser = argparse.ArgumentParser(
        description="Sudoku Solver Comparison– Mia Shuteva 89231346"
    )
    parser.add_argument(
        "csv_path",
        help="Path to sudoku CSV file (format: unsolved,solved per line, no header)"
    )
    parser.add_argument(
        "--samples", type=int, default=500,
        help="Number of puzzles to use (default: 500)"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of ANN training epochs (default: 3)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" Sudoku Solver Comparison")
    print(" Mia Shuteva – 89231346")
    print("=" * 60)

    #load users input and run the pipeline
    summary = run_pipeline(
        csv_path=args.csv_path,
        n_samples=args.samples,
        ann_epochs=args.epochs,
    )

    
    analyse(summary)

    print("\n" + "=" * 60)
    print(" Pipeline complete.")
    print(" results_summary.csv  – numeric results table")
    print(" figures/             – all generated plots")
    print("=" * 60)


if __name__ == "__main__":
    main()
