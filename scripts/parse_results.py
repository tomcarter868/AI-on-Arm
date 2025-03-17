import re
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Check if a file argument is provided
if len(sys.argv) != 2:
    print("Usage: python script.py <results_file>")
    sys.exit(1)

results_file = sys.argv[1]

# Load the results from the specified file
with open(results_file, "r") as f:
    lines = f.readlines()

# Extract relevant data using regex
data = []
pattern = re.compile(r"\|\s+([\w\s]+)\s+\|\s+([\d.]+)\s+GiB\s+\|\s+([\d.]+)\s+B\s+\|\s+CPU\s+\|\s+(\d+)\s+\|\s+(\w+)\s+\|\s+([\d.]+)\s+±\s+([\d.]+)\s+\|")
for line in lines:
    match = pattern.search(line)
    if match:
        model, size, params, threads, test, speed, error = match.groups()
        data.append([model.strip(), float(size), float(params), int(threads), test, float(speed), float(error)])

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Model", "Size (GiB)", "Params (B)", "Threads", "Test", "Speed (t/s)", "Error (t/s)"])



output_file = results_file.replace('.txt', '.csv')
df.to_csv(output_file, index=False)
