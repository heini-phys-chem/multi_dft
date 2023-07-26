#!/usr/bin/env python3

import os, sys
import glob
import re
from cclib.io import ccopen

def extract_single_point_energy(file_path):
    data = ccopen(file_path).parse()
    if hasattr(data, "scfenergies") and data.scfenergies is not None:
        return data.scfenergies[-1]  # Return the last SCF energy
    else:
        return None

def convert_wall_time_to_minutes(wall_time_str):
    hours, minutes, seconds = map(float, wall_time_str.split(":"))
    total_minutes = hours * 60 + minutes + seconds / 60
    return total_minutes

def extract_wall_time(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        matches = re.findall(r'Psi4 wall time for execution: (\d+:\d+:\d+\.\d+)', content)
        if matches:
            return convert_wall_time_to_minutes(matches[-1])
        else:
            return None

def main():
    folder_path = sys.argv[1]
    dat_files = glob.glob(os.path.join(folder_path, "*.dat"))

    for file_path in dat_files:
        single_point_energy = extract_single_point_energy(file_path)
        wall_time = extract_wall_time(file_path)

        if single_point_energy is not None:
            print(os.path.basename(file_path), single_point_energy, wall_time)


if __name__ == "__main__":
    main()
