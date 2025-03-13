PROJECT_NAME = oceanbench

ENVIRONMENT_NAME = ${PROJECT_NAME}
ENVIRONMENT_FILE_NAME = conda_environment.yaml
.ONESHELL:
.SHELLFLAGS = -ec
SHELL := /bin/bash

MICROMAMBA_ACTIVATE=eval "$$(micromamba shell hook --shell=bash)" && micromamba activate
ACTIVATE_ENVIRONMENT=${MICROMAMBA_ACTIVATE} ${ENVIRONMENT_NAME}

_create-update-environment:
	export CONDARC=.condarc
	export PIP_CONFIG_FILE=pip.conf
	(micromamba env update --file ${SELECTED_ENVIRONMENT_FILE_NAME} --name ${SELECTED_ENVIRONMENT_NAME} \
		|| micromamba update --file ${SELECTED_ENVIRONMENT_FILE_NAME} --name ${SELECTED_ENVIRONMENT_NAME} \
		|| micromamba env create --file ${SELECTED_ENVIRONMENT_FILE_NAME} --name ${SELECTED_ENVIRONMENT_NAME})

create-environment: SELECTED_ENVIRONMENT_NAME = ${ENVIRONMENT_NAME}
create-environment: SELECTED_ENVIRONMENT_FILE_NAME = ${ENVIRONMENT_FILE_NAME}
create-environment: _create-update-environment
	micromamba run --name ${ENVIRONMENT_NAME} poetry install

check-format: SELECTED_ENVIRONMENT_NAME = ${ENVIRONMENT_NAME}
check-format:
	${ACTIVATE_ENVIRONMENT}
	pre-commit install
	pre-commit run --all-files --show-diff-on-failure

update-readme: SELECTED_ENVIRONMENT_NAME = ${ENVIRONMENT_NAME}
update-readme:
	${ACTIVATE_ENVIRONMENT}
	jupyter nbconvert --ClearMetadataPreprocessor.enabled=True --ClearOutput.enabled=True --to markdown assets/glonet-example.ipynb
	lead="<!-- BEGINNING of a block automatically generated with make update-readme -->"
	tail="<!-- END of a block automatically generated with make update-readme -->"
	sed -i -e "/^$${lead}/,/^$${tail}/{ /^$${lead}/{p; r assets/glonet-example.md
	}; /^$${tail}/p; d }" README.md
	rm assets/glonet-example.md

test: SELECTED_ENVIRONMENT_NAME = ${ENVIRONMENT_NAME}
test:
	${ACTIVATE_ENVIRONMENT}
	python tests/glonet_sample_evaluation.py
