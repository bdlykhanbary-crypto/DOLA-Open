from pathlib import Path
import csv
import math
import os

DATA = str(Path(__file__).resolve().parent / "data")

features = [
    "return_1", "return_2", "return_3", "return_5",
    "return_7", "return_14", "return_21", "return_30",
    "return_45", "return_63",
    "volatility_7", "volatility_14", "volatility_30",
    "volatility_60", "daily_range", "open_gap",
    "close_position", "weekday", "month"
]

target = "target_return_30d"

for part in ["train", "validation", "test"]:
    source = os.path.join(DATA, part + "_purged.csv")
    output = os.path.join(DATA, part + "_30d.libsvm")

    count = 0

    with open(source, newline="") as fin, open(output, "w") as fout:
        reader = csv.DictReader(fin)

        for row in reader:
            y = float(row[target])
            values = [float(row[name]) for name in features]

            if not math.isfinite(y) or not all(map(math.isfinite, values)):
                raise ValueError("Invalid numeric value in " + part)

            line = [format(y, ".17g")]

            for i, value in enumerate(values, start=1):
                line.append(f"{i}:{value:.17g}")

            fout.write(" ".join(line) + "\n")
            count += 1

    print(part, ":", count, "samples converted")

print("All datasets converted successfully.")
