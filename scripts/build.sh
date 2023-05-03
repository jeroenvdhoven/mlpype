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
    packages=($(ls pype))
fi

python -m build . --outdir $output_dir

# Build all requested packages.
for package in "${packages[@]}"
do
    package_path="pype/${package}"
    echo "Building: ${package_path}"
    # install_package $package_path $host $editable_string
    python -m build $package_path --outdir $output_dir
done