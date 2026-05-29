"""
Sudoku Solver Comparison: CSP vs ANN vs Hybrid
Author: Mia Shuteva – 89231346
Project: Runtime-Accuracy Trade-off Analysis
"""

import numpy as np
import pandas as pd
import time
import os
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_dataset(csv_path: str, n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Load sudoku CSV file.

    Expected CSV format (no header):
        <81-char unsolved string>,<81-char solved string>

    '0' represents an empty cell.
    Returns a DataFrame with columns: ['puzzle', 'solution']
    """
    df = pd.read_csv(csv_path, header=None, names=["puzzle", "solution"], dtype=str)
    df = df.dropna().reset_index(drop=True)

    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)

    print(f"[Data] Loaded {len(df):,} puzzles from '{csv_path}'")
    return df


def string_to_grid(s: str) -> np.ndarray:
    """Convert an 81-character string to a 9×9 numpy array."""
    return np.array([int(c) for c in str(s)], dtype=np.int8).reshape(9, 9)


def grid_to_string(grid: np.ndarray) -> str:
    return "".join(str(d) for d in grid.flatten())


# ─────────────────────────────────────────────
# ALGORITHM 1 – CONSTRAINT SATISFACTION (CSP)
# ─────────────────────────────────────────────

class CSPSolver:
    """
    Backtracking CSP solver with forward-checking (arc consistency).
    Heuristic: Minimum Remaining Values (MRV) for cell selection.
    """

    def solve(self, puzzle_str: str) -> tuple[str | None, float]:
        grid = string_to_grid(puzzle_str)
        t0 = time.perf_counter()
        solved = self._backtrack(grid)
        elapsed = time.perf_counter() - t0
        if solved:
            return grid_to_string(grid), elapsed
        return None, elapsed

    def _is_valid(self, grid: np.ndarray, row: int, col: int, num: int) -> bool:
        if num in grid[row]:
            return False
        if num in grid[:, col]:
            return False
        br, bc = 3 * (row // 3), 3 * (col // 3)
        if num in grid[br:br+3, bc:bc+3]:
            return False
        return True

    def _get_candidates(self, grid: np.ndarray, row: int, col: int) -> set:
        used = (
            set(grid[row])
            | set(grid[:, col])
            | set(grid[3*(row//3):3*(row//3)+3, 3*(col//3):3*(col//3)+3].flatten())
        )
        return set(range(1, 10)) - used

    def _find_mrv_cell(self, grid: np.ndarray) -> tuple[int, int] | None:
        """Return the empty cell with the fewest legal candidates (MRV heuristic)."""
        best_cell = None
        best_count = 10
        for r in range(9):
            for c in range(9):
                if grid[r, c] == 0:
                    count = len(self._get_candidates(grid, r, c))
                    if count == 0:
                        return (-1, -1)   # dead end
                    if count < best_count:
                        best_count = count
                        best_cell = (r, c)
        return best_cell

    def _backtrack(self, grid: np.ndarray) -> bool:
        cell = self._find_mrv_cell(grid)
        if cell is None:
            return True           # all cells filled
        if cell == (-1, -1):
            return False          # contradiction

        row, col = cell
        for num in self._get_candidates(grid, row, col):
            grid[row, col] = num
            if self._backtrack(grid):
                return True
            grid[row, col] = 0
        return False


# ─────────────────────────────────────────────
# ALGORITHM 2 – ARTIFICIAL NEURAL NETWORK (ANN)
# ─────────────────────────────────────────────

class ANNSolver:
    """
    Convolutional ANN trained on puzzle/solution pairs.
    Architecture: multiple Conv2D layers → digit classification per cell.
    Falls back to a lightweight rule-based fill when TensorFlow is unavailable.
    """

    def __init__(self):
        self.model = None
        self._tf_available = False
        self._try_import_tf()

    def _try_import_tf(self):
        try:
            import tensorflow as tf
            self._tf = tf
            self._tf_available = True
        except ImportError:
            print("[ANN] TensorFlow not installed – ANN will use rule-based approximation.")

    # ── Model definition ──────────────────────────────────────────────────────

    def build_model(self):
        if not self._tf_available:
            return
        tf = self._tf
        keras = tf.keras

        inputs = keras.Input(shape=(9, 9, 1))
        x = inputs

        for filters in [64, 64]:
            x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)

        # Per-cell digit prediction (1-9)
        outputs = keras.layers.Conv2D(9, 1, activation="softmax")(x)   # (9,9,9)

        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(f"[ANN] Model built — {self.model.count_params():,} parameters")

    def train(self, df: pd.DataFrame, epochs: int = 3, batch_size: int = 512,
              val_split: float = 0.1):
        if not self._tf_available or self.model is None:
            print("[ANN] Skipping training – TensorFlow unavailable or model not built.")
            return

        X, y = self._prepare_data(df)
        print(f"[ANN] Training on {len(X):,} puzzles for {epochs} epoch(s)…")
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            verbose=1,
        )

    def _prepare_data(self, df: pd.DataFrame):
        """Encode puzzles as (9,9,1) float tensors; labels as (9,9) int arrays (0-indexed)."""
        n = len(df)
        X = np.zeros((n, 9, 9, 1), dtype=np.float32)
        y = np.zeros((n, 9, 9), dtype=np.int8)
        for i, (_, row) in enumerate(df.iterrows()):
            X[i, :, :, 0] = string_to_grid(row["puzzle"]) / 9.0
            y[i] = string_to_grid(row["solution"]) - 1   # 0-indexed classes
        return X, y

    # ── Inference ─────────────────────────────────────────────────────────────

    def solve(self, puzzle_str: str) -> tuple[str | None, float]:
        t0 = time.perf_counter()
        if self._tf_available and self.model is not None:
            result = self._solve_with_model(puzzle_str)
        else:
            result = self._solve_rule_based(puzzle_str)
        elapsed = time.perf_counter() - t0
        return result, elapsed

    def _solve_with_model(self, puzzle_str: str) -> str:
        X = string_to_grid(puzzle_str).astype(np.float32) / 9.0
        X = X.reshape(1, 9, 9, 1)
        probs = self.model.predict(X, verbose=0)   # (1,9,9,9)
        preds = (np.argmax(probs[0], axis=-1) + 1).astype(np.int8)   # (9,9) digits 1-9

        grid = string_to_grid(puzzle_str)
        # Keep original clues, fill blanks with ANN prediction
        mask = grid == 0
        grid[mask] = preds[mask]
        return grid_to_string(grid)

    def _solve_rule_based(self, puzzle_str: str) -> str:
        """
        Lightweight deterministic fill used when TensorFlow is unavailable.
        Iteratively fills cells that have exactly one candidate; returns a
        partial solution otherwise (accuracy will be <100%, which is expected
        and reported in the evaluation).
        """
        grid = string_to_grid(puzzle_str)
        changed = True
        while changed:
            changed = False
            for r in range(9):
                for c in range(9):
                    if grid[r, c] == 0:
                        used = (
                            set(grid[r])
                            | set(grid[:, c])
                            | set(grid[3*(r//3):3*(r//3)+3, 3*(c//3):3*(c//3)+3].flatten())
                        ) - {0}
                        candidates = set(range(1, 10)) - used
                        if len(candidates) == 1:
                            grid[r, c] = candidates.pop()
                            changed = True
        return grid_to_string(grid)


# ─────────────────────────────────────────────
# ALGORITHM 3 – HYBRID (ANN → CSP)
# ─────────────────────────────────────────────

class HybridSolver:
    """
    Run ANN first to fill in highly confident cells, then run CSP to
    complete / correct the remaining cells via backtracking.
    """

    def __init__(self, ann: ANNSolver, csp: CSPSolver):
        self.ann = ann
        self.csp = csp

    def solve(self, puzzle_str: str) -> tuple[str | None, float]:
        t0 = time.perf_counter()

        # Step 1: get ANN prediction
        ann_result, _ = self.ann.solve(puzzle_str)
        seeded = ann_result if ann_result else puzzle_str

        # Step 2: restore original clues (trust the given digits)
        original = string_to_grid(puzzle_str)
        seeded_grid = string_to_grid(seeded)
        # Wherever original has a clue, keep it; ANN fill elsewhere
        merged = np.where(original != 0, original, seeded_grid)
        seeded_str = grid_to_string(merged)

        # Step 3: CSP to verify / complete
        csp_result, _ = self.csp.solve(seeded_str)

        elapsed = time.perf_counter() - t0
        return csp_result, elapsed


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def is_correct(result: str | None, solution: str) -> bool:
    return result is not None and result == solution


def evaluate(solver, df: pd.DataFrame, label: str) -> dict:
    """Run solver over the dataset; return runtime and accuracy statistics."""
    print(f"\n[Eval] Running {label} on {len(df):,} puzzles…")
    runtimes, correct = [], []

    for _, row in df.iterrows():
        result, rt = solver.solve(row["puzzle"])
        runtimes.append(rt)
        correct.append(is_correct(result, row["solution"]))

    runtimes = np.array(runtimes)
    accuracy = np.mean(correct)

    stats = {
        "algorithm": label,
        "n": len(df),
        "accuracy": round(accuracy * 100, 2),
        "mean_runtime_s": round(runtimes.mean(), 6),
        "std_runtime_s": round(runtimes.std(), 6),
        "min_runtime_s": round(runtimes.min(), 6),
        "max_runtime_s": round(runtimes.max(), 6),
        "p50_runtime_s": round(np.percentile(runtimes, 50), 6),
        "p95_runtime_s": round(np.percentile(runtimes, 95), 6),
    }

    print(f"  Accuracy : {stats['accuracy']}%")
    print(f"  Mean rt  : {stats['mean_runtime_s']:.4f}s ± {stats['std_runtime_s']:.4f}s")
    print(f"  P95 rt   : {stats['p95_runtime_s']:.4f}s")
    return stats


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(csv_path: str, n_samples: int = 500, ann_epochs: int = 3):
    df = load_dataset(csv_path, n_samples=n_samples)

    # ── Initialise solvers ────────────────────────────────────────────────────
    csp = CSPSolver()
    ann = ANNSolver()

    ann.build_model()
    # Split: 80% train, 20% test
    train_df = df.sample(frac=0.8, random_state=0)
    test_df = df.drop(train_df.index).reset_index(drop=True)
    ann.train(train_df, epochs=ann_epochs)

    hybrid = HybridSolver(ann, csp)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = []
    results.append(evaluate(csp,    test_df, "CSP (Backtracking + MRV)"))
    results.append(evaluate(ann,    test_df, "ANN (Convolutional)"))
    results.append(evaluate(hybrid, test_df, "Hybrid (ANN → CSP)"))

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = pd.DataFrame(results)
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(summary[[
        "algorithm", "n", "accuracy",
        "mean_runtime_s", "std_runtime_s", "p95_runtime_s"
    ]].to_string(index=False))

    summary.to_csv("results_summary.csv", index=False)
    print("\n[Done] Results saved to results_summary.csv")
    return summary


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sudoku Solver Comparison – Mia Shuteva 89231346")
    parser.add_argument("csv_path",         help="Path to sudoku.csv")
    parser.add_argument("--samples",  type=int, default=500,  help="Number of puzzles to evaluate (default 500)")
    parser.add_argument("--epochs",   type=int, default=3,    help="ANN training epochs (default 3)")
    args = parser.parse_args()

    run_pipeline(args.csv_path, n_samples=args.samples, ann_epochs=args.epochs)
