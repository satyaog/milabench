# /// script
# dependencies = [
#   "jinja2",
#   "matplotlib",
#   "pandas",
#   "scikit-learn",
# ]
# [tool.hatch]
# python = "3.10"
# python-sources = ["external"]
# installer = "uv"
# ///

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.DEBUG)

# Change display settings to show the entire table
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # No line width limit
pd.set_option('display.max_colwidth', None)  # Show full content of each cell

GPUS = {
    "NVIDIA-A100-SXM4-40GB": {"fp16":311.869440, "fp32":19.491840, "fp64": 9.745920, "tf32":155.934720, "memgb":40,  "membw":1555, "tdp":400},
    "NVIDIA-A100-SXM4-80GB": {"fp16":311.869440, "fp32":19.491840, "fp64": 9.745920, "tf32":155.934720, "memgb":80,  "membw":1555, "tdp":400},
    "NVIDIA-H100-80GB-HBM3": {"fp16":989.429760, "fp32":66.908160, "fp64":33.454080, "tf32":494.714880, "memgb":80,  "membw":3352, "tdp":700},
    "NVIDIA-L40S"          : {"fp16":366.428160, "fp32":91.607040, "fp64": 1.431360, "tf32":183.214080, "memgb":48,  "membw": 864, "tdp":350},
    "Quadro-RTX-8000"      : {"fp16":130.498560, "fp32":16.312320, "fp64": 0.509760, "tf32":       0.0, "memgb":48,  "membw": 672, "tdp":260},
    "Tesla-V100-SXM2-32GB" : {"fp16":125.337600, "fp32":15.667200, "fp64": 7.833600, "tf32":       0.0, "memgb":32,  "membw": 900, "tdp":300},
}
REF_GPU = "NVIDIA-A100-SXM4-80GB"

GPUS = pd.DataFrame(GPUS).T
logging.debug(f"\n{GPUS}")
GPUS = GPUS / GPUS.loc[REF_GPU]
GPUS = GPUS.drop(columns=[
    "fp64",
    "tf32",
    "tdp"
])
logging.info(f"\n{GPUS}")

df = pd.read_csv(str(Path("log.csv")))
df.set_index(["gpu", "bench", "batch_size"], inplace=True)

# drop weight == 0
df = df[df["weight"] != 0]

# convert str
df["sem%"] = [float(sem[:-1]) / 100 for sem in df["sem%"]]
df["std%"] = [float(std[:-1]) / 100 for std in df["std%"]]

gpu_aliases = {
    "NVIDIA-A100-SXM4-40GB": ["a100"],
    "NVIDIA-A100-SXM4-80GB": ["a100l"],
    "NVIDIA-H100-80GB-HBM3": [],
    "NVIDIA-L40S": [],
    "Quadro-RTX-8000": [],
    "Tesla-V100-SXM2-32GB": ["Tesla-V100-SXM2-32GB-LS", "v100"],
}

for gpu, aliases in gpu_aliases.items():
    for alias in aliases:
        aliases_data = df[df.index.get_level_values("gpu") == alias].copy(deep=True)

        df.drop(aliases_data.index, inplace=True)
        aliases_data.reset_index(inplace=True)
        aliases_data.loc[:, "gpu"] = gpu
        aliases_data.set_index(["gpu", "bench", "batch_size"], inplace=True)
        aliases_data.drop(
            aliases_data.index[
                [i in df.index for i in aliases_data.index]
            ],
            inplace=True
        )

        df = pd.concat([df, aliases_data])

df = df.sort_index(level=2).sort_index(level=1).sort_index(level=0)

for bench in sorted(df.index.get_level_values("bench").unique()):
    logging.info(f"{bench}:\n{df.loc[:, bench, :]}")

missing_benches = set()
valid_benches = set(df.index.get_level_values("bench").unique())

for bench in sorted(df.index.get_level_values("bench").unique()):
    for gpu in df.index.get_level_values("gpu").unique():
        bench_data = df.loc[gpu, bench, :]
        df.drop(
            [(gpu, bench, _idx) for _idx in bench_data.index],
            inplace=True
        )

        if bench == "brax":
            bench_data = bench_data[bench_data["ngpu"] == 1]
            logging.warning(f"{(gpu, bench)}: Dropped entries with more than 1 gpu\n{bench_data}")

        if bench == "lightning-gpus":
            bench_data = bench_data[bench_data["ngpu"] == 4]
            logging.warning(f"{(gpu, bench)}: Dropped entries with less than 4 gpu\n{bench_data}")

        if bench == "ppo":
            bench_data = bench_data[bench_data.index.get_level_values("batch_size") != 7808]
            logging.warning(f"{(gpu, bench)}: Dropped {bench} entry with a batch size of 7808\n{bench_data}")

        if bench_data.empty or bench_data["perf"].isna().all():
            logging.warning(f"Missing bench data for {(gpu, bench)}")
            if bench in valid_benches:
                valid_benches.remove(bench)

            missing_benches.add(bench)
            missing_bench = pd.Series(
                {
                    "gpu": gpu,
                    "bench": bench,
                    "batch_size": 0,
                    "weight": df.loc[:, bench, :]["weight"].max(),
                },
            )
            df = pd.concat(
                [
                    df,
                    missing_bench.to_frame().T.set_index(["gpu", "bench", "batch_size"])
                ]
            )
            continue

        if len(bench_data.loc[bench_data["perf"].idxmax()].shape) > 1:
            best = bench_data.loc[bench_data["perf"].idxmax()].mean()
        else:
            best = bench_data.loc[bench_data["perf"].idxmax()].copy()
        best["gpu"] = gpu
        best["bench"] = bench
        best["batch_size"] = bench_data["perf"].idxmax()

        df = pd.concat(
            [
                df,
                best.to_frame().T.set_index(["gpu", "bench", "batch_size"])
            ]
        )

df.sort_index(level=2, inplace=True)
df.sort_index(level=1, inplace=True)
df.sort_index(level=0, inplace=True)

print(df)
df = df.infer_objects()

def _score(_df:pd.DataFrame):
    weights = _df["weight"]
    sum_weights = weights.sum()
    weights[_df["perf"].isna()] = 0.0
    return (
        (np.log(_df["perf"]) * weights).sum() / sum_weights
    )

scores = df.groupby("gpu").apply(_score)
logging.debug(f"log\n{scores}")
scores = np.exp(scores)
logging.debug(f"exp(log)\n{scores}")
scores = scores / scores.loc[REF_GPU]
print(scores)
print()

# Features (independent variables)
X = GPUS

# Target variable (dependent variable)
y = scores

# Create a Linear Regression model
model = LinearRegression(fit_intercept=True)

# Fit the model on the data
model.fit(X, y)

# Coefficients (beta values)
coefficients = model.coef_

# Intercept (beta_0)
intercept = model.intercept_

# Print the results
print("Intercept (β₀):", intercept)
print("Coefficients (β₁, β₂, ... β₆):", pd.DataFrame(
    [
        coefficients,
        coefficients / sum(coefficients)
    ],
    index=["raw", "sum to 1"],
    columns=X.columns.tolist()
), sep="\n")

coefficients_floor = coefficients[coefficients < 0.05] = 0.05

print(pd.DataFrame(
    [
        coefficients,
        coefficients / sum(coefficients)
    ],
    index=["raw", "sum to 1"],
    columns=X.columns.tolist()
))

out = {
    "pred": [coefficients.dot(X.loc[i]) + intercept for i in X.index],
}
out = pd.DataFrame(out, index = X.index)

print(out)
