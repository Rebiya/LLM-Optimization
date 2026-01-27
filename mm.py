from pathlib import Path

dirs = [
    "data/mnist",
    "models",
    "notebooks",
    "experiments",
    "report",
]

files = [
    "models/ffn_baseline.pth",
    "models/moe_ffn.pth",
    "models/quantized_model.pth",
    "models/student_model.pth",
    "notebooks/01_baseline.ipynb",
    "notebooks/02_moe.ipynb",
    "notebooks/03_ptq.ipynb",
    "notebooks/04_kd.ipynb",
    "experiments/results.csv",
    "report/report.md",
    "README.md",
]

for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)

for f in files:
    Path(f).touch(exist_ok=True)

print("Folder structure created successfully ✅")
