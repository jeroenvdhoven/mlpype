#! /bin/bash

# Look for output-dir and packages flags.
while getopts "o:p:" flag
do
    case "${flag}" in
        h) output_dir=${OPTARG};;
        e) packages=${OPTARG};;
    esac
done
output_dir="${editable:-dist}"
if [ -z $packages ]
then
    packages=($(ls . | grep mlpype- ))
fi

python -m build . --outdir $output_dir

# Build all requested packages.
for package in "${packages[@]}"
do
    echo "Building: ${package}"
    # install_package $package $host $editable_string
    python -m build $package --outdir $output_dir
done