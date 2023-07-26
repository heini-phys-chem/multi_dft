#!/usr/bin/env python3

import sys
import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file into a DataFrame
csv_file1 = sys.argv[1]
csv_file2 = sys.argv[2]
df1 = pd.read_csv(csv_file1, names=["Filename", "Energy", "Time (minutes)"])
df2 = pd.read_csv(csv_file2, names=["Filename", "Energy", "Time (minutes)"])

# Create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot histogram of energies
ax1.hist(df1["Energy"], bins=20,  alpha=0.5, color="C0", label="PBE")
ax1.hist(df2["Energy"], bins=20,  alpha=0.5, color="C1", label="PBEP86")
ax1.set_title("Distribution of Energies")
ax1.set_xlabel("Energy")
ax1.set_ylabel("Frequency")

ax1.legend()

# Plot histogram of timings
ax2.hist(df1["Time (minutes)"], bins=20,  alpha=0.2, color="C0")
ax2.hist(df2["Time (minutes)"], bins=20,  alpha=0.2, color="C1")
ax2.set_title("Distribution of Timings")
ax2.set_xlabel("Time (minutes)")
ax2.set_ylabel("Frequency")

plt.tight_layout()
plt.show()

