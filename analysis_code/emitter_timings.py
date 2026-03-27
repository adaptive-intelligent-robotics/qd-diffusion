import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
BASE_PATH = Path("/workspace/src/hpc_results/timing_results/sdxl-turbo") # Replace with your actual PATH
EXP_NUMS = [1, 4, 7, 10, 13, 16, 21, 22, 25, 28, 31, 34]
# Dictionary of algorithms: { "folder_name": "Display Name for Plot" }
ALGORITHMS = {
    "multi-hidden-layer-multi-svd": "multi-hidden-layer-svd",
    "mixed-all": "mixed-all",
    "multi-adam": "multi-adam",
    "multi-gaussian": "multi-gaussian",
    "multi-hidden-layer-gaussian": "multi-hidden-layer-gaussian",
    "multi-isoline": "multi-isoline",
    "multi-sep-cma-es": "multi-sep-cma-es",
    "random-seeds": "random-seeds"
}

def collect_data():
    all_data = []

    for algo_folder, legend_name in ALGORITHMS.items():
        for num in EXP_NUMS:
            # Use glob to handle the unknown EXP_DATE directory
            search_pattern = f"{num}/dns/{algo_folder}/*/metrics/mapelites-logs.csv"
            files = list(BASE_PATH.glob(search_pattern))
            
            if not files:
                print(f"Warning: No file found for {algo_folder} Exp {num}")
                continue
            
            # Assuming one file per experiment; take the first one found
            file_path = files[0]
            
            try:
                df = pd.read_csv(file_path)
                if "emitter_time" in df.columns:
                    # Get the final value in the emitter_time column
                    final_time = df["emitter_time"].iloc[-1]
                    all_data.append({
                        "Algorithm": legend_name,
                        "Time": final_time,
                        "Exp": num
                    })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return pd.DataFrame(all_data)

def plot_timings(df):
    if df.empty:
        print("No data collected. Check your paths!")
        return

    plt.figure(figsize=(10, 6))
    
    # Create the boxplot
    ax = plt.boxplot([df[df["Algorithm"] == algo]["Time"] for algo in df["Algorithm"].unique()],
                     labels=df["Algorithm"].unique(), patch_artist=True)
    # Optional: Add jittered points to see individual experiment results
    for i, algo in enumerate(df["Algorithm"].unique()):
        y = df[df["Algorithm"] == algo]["Time"]
        print("Algorithm:", algo)
        print("Median time:", y.median()/2000, "seconds")
        print("----")
        x = np.random.normal(i + 1, 0.04, size=len(y))
        plt.scatter(x, y, color="black", alpha=0.3)

    plt.title("Emitter Timing Distribution per Algorithm")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Algorithm")
    
    # Save the output
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "boxplot_timings.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    results_df = collect_data()
    plot_timings(results_df)