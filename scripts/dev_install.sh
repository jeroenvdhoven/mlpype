#! /bin/bash

function install_package () {
    # 0: script name
    # 1: package name
    # 2: host
    # 3: editable. Ignored if host is set.
    raw_name=$1
    package="${raw_name/.//}"

    echo "host: $2"
    
    if [ $2 != "local" ];
    then
        echo "Installing $1 from host $2"
        ip=`echo $2 or grep -oE [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+`
        echo "IP: $ip"
        pip install -i $2 $1 --trusted-host $ip --upgrade
    else
        echo "Installing ${package} from local machine"
        pip install $3 "${package}[dev]"  --upgrade
    fi;
}

# Look for editable and host flags.
while getopts "e:h:" flag
do
    case "${flag}" in
        h) host=${OPTARG};;
        e) editable=${OPTARG};;
    esac
done
host="${host:-local}"
editable="${editable:-1}"

echo "Editable state: $editable, host: $host"

# Set editable
if [ $editable == 1 ];
then
    editable_string=" -e "
else
    editable_string=""
fi;

# Install priority packages first: base and sklearn. These are packages that others depend on.
# Please note, that especially tensorflow may run into issues. On some Mac machines (M1 versions)
# the result installation may fail. It is recommended in those cases to first manually install
# tensorflow, then install all pype packages.
priority_packages=( "pype.base" "pype.sklearn" )
for priority_package in "${priority_packages[@]}"
do
    echo "Priority installing: ${priority_package}"
    install_package $priority_package $host $editable_string
done

# Install all remaining packages.
packages=($(ls pype))
for package in "${packages[@]}"
do
    package_path="pype.${package}"
    if [[ ! " ${priority_packages[*]} " =~ " ${package_path} " ]]; then
        # only install if package wasn't priority package
        echo "Installing: ${package_path}"
        install_package $package_path $host $editable_string
    fi
done