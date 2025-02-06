PROJECT_NAME = oceanbench

ENVIRONMENT_NAME = ${PROJECT_NAME}
ENVIRONMENT_FILE_NAME = conda_environment.yaml
.ONESHELL:
.SHELLFLAGS = -ec
SHELL := /bin/bash

MICROMAMBA_ACTIVATE=eval "$$(micromamba shell hook --shell=bash)" && micromamba activate
ACTIVATE_ENVIRONMENT=${MICROMAMBA_ACTIVATE} ${ENVIRONMENT_NAME}

create-update-environment:
	export CONDARC=.condarc
	export PIP_CONFIG_FILE=pip.conf
	(micromamba env update --file ${SELECTED_ENVIRONMENT_FILE_NAME} --name ${SELECTED_ENVIRONMENT_NAME} \
		|| micromamba update --file ${SELECTED_ENVIRONMENT_FILE_NAME} --name ${SELECTED_ENVIRONMENT_NAME} \
		|| micromamba env create --file ${SELECTED_ENVIRONMENT_FILE_NAME} --name ${SELECTED_ENVIRONMENT_NAME})

create-environment: SELECTED_ENVIRONMENT_NAME = ${ENVIRONMENT_NAME}
create-environment: SELECTED_ENVIRONMENT_FILE_NAME = ${ENVIRONMENT_FILE_NAME}
create-environment: create-update-environment

check-format: SELECTED_ENVIRONMENT_NAME = ${ENVIRONMENT_NAME}
check-format:
	${ACTIVATE_ENVIRONMENT}
	pre-commit install
	pre-commit run --all-files --show-diff-on-failure
