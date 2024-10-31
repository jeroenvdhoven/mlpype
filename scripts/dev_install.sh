#! /bin/bash


function install_packages () {
    # 0: script name
    # 1: host
    # 2: editable. Ignored if host is set.
    # 3+: packages
    host=$1
    editable=$2
    packages="${@:3}"    

    if [ ${host} != "local" ];
    then
        package_str=""
        for package in "${packages[@]}"
        do
            package_str="${package_str} ${package}"
        done

        echo "Installing ${package_str} from host ${host}"
        ip=`echo ${host} | grep -oE [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+`
        echo "IP: $ip"
        
        # pip install --index-url ${host} ${package_str} --trusted-host $ip --upgrade
    else
        # make sure we use the right path for local installation
        echo "Installing from local machine: ${packages[@]}"
        package_str=""
        for package in "${packages[@]}"
        do
            package_path="./${package/.//}"
            if [ $editable == 1 ];
            then
                package_path="-e ${package_path}[dev,strict]"
            else
                package_path="${package_path}[dev,strict]"
            fi;
            package_str="${package_str} ${package_path}"
        done

        echo "pip install ${package_str}  --upgrade"
        pip install ${package_str}  --upgrade
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
editable="${editable:-0}"

echo "Editable state: $editable, host: $host"

# Install priority packages first: base and sklearn. These are packages that others depend on.
# Please note, that especially tensorflow may run into issues. On some Mac machines (M1 versions)
# the installation may fail. It is recommended in those cases to first manually install
# tensorflow, then install all mlpype packages.
packages=( $(ls . | grep mlpype- ) )
echo "Installing: ${packages[@]}"
install_packages ${host} ${editable} "${packages}" 
