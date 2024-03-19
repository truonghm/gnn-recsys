#!/bin/bash

set -e

PROJECT_ENV_LOC="conda"
ENV_NAME="gnn-recsys"
PYTHON_VERSION="3.10"
INSTALL_CONDA=false
CONDA_HOME_PATH="${HOME}/miniconda3"
REQUIREMENTS_PATH="requirements.txt"

TEMP=`getopt -o l:n:v:r: --long location:,name:,version:,requirements:,install-conda,no-install-conda -n 'conda_bootstrap.sh' -- "$@"`
eval set -- "$TEMP"

while true ; do
    case "$1" in
        --location)
            PROJECT_ENV_LOC=$2 ; shift 2 ;;
        --name)
            ENV_NAME=$2 ; shift 2 ;;
        --version)
            PYTHON_VERSION=$2 ; shift 2 ;;
        --requirements)
            REQUIREMENTS_PATH=$2 ; shift 2 ;;
        --install-conda)
            INSTALL_CONDA=true ; shift ;;
        --no-install-conda)
            INSTALL_CONDA=false ; shift ;;
        --) shift ; break ;;
        *) echo "Error: Unexpected option or argument encountered. Please check your input and try again." ; exit 1 ;;
    esac
done

echo "PROJECT_ENV_LOC: ${PROJECT_ENV_LOC}"
echo "ENV_NAME: ${ENV_NAME}"
echo "PYTHON_VERSION: ${PYTHON_VERSION}"
echo "INSTALL_CONDA: ${INSTALL_CONDA}"

if $INSTALL_CONDA; then
    echo "Installing conda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${HOME}/miniconda.sh
    bash ${HOME}/miniconda.sh -b -p ${CONDA_HOME_PATH}
    rm ${HOME}/miniconda.sh
    source ${CONDA_HOME_PATH}/bin/activate; conda init bash; conda config --set auto_activate_base false
fi

if [ ${PROJECT_ENV_LOC} == "current" ]; then
	echo "Creating conda environment in current directory..."
	CONDA_ENV_PARENT_PATH=".conda"
	CONDA_ENV_PATH="${CONDA_ENV_PARENT_PATH}/${ENV_NAME}"

	rm -rf ${CONDA_ENV_PATH} || true

	source ${CONDA_HOME_PATH}/bin/activate; conda create -p ${CONDA_ENV_PATH} --no-default-packages --no-deps python=${PYTHON_VERSION} -y; conda install -p ${CONDA_ENV_PATH} anaconda::pip -y
	touch ${CONDA_ENV_PATH}/.gitignore
	echo "*" > ${CONDA_ENV_PATH}/.gitignore

	# echo "Installing requirements..."

	# source ${CONDA_HOME_PATH}/bin/activate ${CONDA_ENV_PATH}; echo $CONDA_PREFIX; pip install -r ${REQUIREMENTS_PATH};

elif [ ${PROJECT_ENV_LOC} == "conda" ]; then

	echo "Creating conda environment in conda directory..."

	source ${CONDA_HOME_PATH}/bin/activate; conda remove --name ${ENV_NAME} --all; conda create --name=${ENV_NAME} --no-default-packages --no-deps python=${PYTHON_VERSION} -y; conda install -n ${ENV_NAME} anaconda::pip -y

	# echo "Installing requirements..."

	# source ${CONDA_HOME_PATH}/bin/activate ${ENV_NAME}; echo $CONDA_PREFIX; pip install -r ${REQUIREMENTS_PATH};
else
	echo "Invalid argument, must be either `current` or `conda`"
	exit 1
fi