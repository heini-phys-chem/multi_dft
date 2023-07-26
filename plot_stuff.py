#!/usr/bin/env python3

import matplotlib.pyplot as plt

def read_run_times(file_path):
    with open(file_path, 'r') as file:
        run_times = [float(line.strip()) for line in file]
    return run_times

def plot_histogram(run_times):
    plt.hist(run_times, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Run Times (minutes)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Run Times')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file_path = "runtime_mins.txt"  # Replace with the actual file path
    run_times = read_run_times(file_path)
    plot_histogram(run_times)

