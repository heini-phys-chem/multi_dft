#!/bin/bash

# Check if the 'xyz' directory exists
if [ ! -d "xyz" ]; then
    echo "Error: Directory 'xyz' not found."
    exit 1
fi

# Create the 'run' directory if it doesn't exist
if [ ! -d "run_LC-wPBE" ]; then
    mkdir "run_LC-wPBE"
fi

# Process each XYZ file in the 'xyz' directory
for xyz_file in xyz/*.xyz; do
    # Get the filename without the extension
    filename_no_ext=$(basename "${xyz_file%.*}")

    # Create a directory in 'run' with the filename as the directory name
#    run_directory="run_PBE0/$filename_no_ext"
#    mkdir "$run_directory"

    # Generate the Psi4 input file content
    psi4_input=$(cat <<EOF
molecule {
  0 1
  $(tail -n +3 "$xyz_file")
}

set basis aug-cc-pvtz
energy('WPBE-D3BJ')
EOF
)

    # Write the Psi4 input file in the corresponding 'run' directory
    echo "$psi4_input" > "run_LC-wPBE/$filename_no_ext.psi4"
    echo "generated $filename_no_ext"
done

echo "Psi4 input files generated in the 'run/' directory."

