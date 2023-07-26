#!/bin/bash

while true; do
    # Count the number of .dat files in the current directory
    count=$(find . -maxdepth 1 -type f -name "*.dat" | wc -l)

    # Clear the line and print the count
    printf "\rNumber of .dat files: $count"

    # Wait for 1 minute before checking again
    sleep 60
done

