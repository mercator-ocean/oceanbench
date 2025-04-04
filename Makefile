PROJECT_NAME = oceanbench

ENVIRONMENT_NAME = ${PROJECT_NAME}
ENVIRONMENT_FILE_NAME = conda_environment.yaml
TEST_ENVIRONMENT_NAME = ${PROJECT_NAME}_test
TEST_ENVIRONMENT_FILE_NAME = conda_environment_test.yaml
.ONESHELL:
.SHELLFLAGS = -ec
SHELL := /bin/bash

MICROMAMBA_ACTIVATE=eval "$$(micromamba shell hook --shell=bash)" && micromamba activate
ACTIVATE_ENVIRONMENT=${MICROMAMBA_ACTIVATE} ${SELECTED_ENVIRONMENT_NAME}

create-update-environment:
	export CONDARC=.condarc
	export PIP_CONFIG_FILE=pip.conf
	(micromamba env update --file ${SELECTED_ENVIRONMENT_FILE_NAME} --name ${SELECTED_ENVIRONMENT_NAME} \
		|| micromamba update --file ${SELECTED_ENVIRONMENT_FILE_NAME} --name ${SELECTED_ENVIRONMENT_NAME} \
		|| micromamba env create --file ${SELECTED_ENVIRONMENT_FILE_NAME} --name ${SELECTED_ENVIRONMENT_NAME})

create-environment: SELECTED_ENVIRONMENT_NAME = ${ENVIRONMENT_NAME}
create-environment: SELECTED_ENVIRONMENT_FILE_NAME = ${ENVIRONMENT_FILE_NAME}
create-environment: _create-update-environment
	micromamba run --name ${ENVIRONMENT_NAME} poetry install

create-test-environment: SELECTED_ENVIRONMENT_NAME = ${TEST_ENVIRONMENT_NAME}
create-test-environment: SELECTED_ENVIRONMENT_FILE_NAME = ${TEST_ENVIRONMENT_FILE_NAME}
create-test-environment: create-update-environment

check-format: SELECTED_ENVIRONMENT_NAME = ${ENVIRONMENT_NAME}
check-format:
	${ACTIVATE_ENVIRONMENT}
	pre-commit install
	pre-commit run --all-files --show-diff-on-failure

update-readme: SELECTED_ENVIRONMENT_NAME = ${ENVIRONMENT_NAME}
update-readme:
	${ACTIVATE_ENVIRONMENT}
	python -c 'import oceanbench; oceanbench.generate_notebook_to_evaluate("assets/glonet_sample.py", "assets/glonet_sample.ipynb")'
	jupyter nbconvert --ClearMetadataPreprocessor.enabled=True --ClearOutput.enabled=True --to markdown assets/glonet_sample.ipynb
	lead="<!-- BEGINNING of a block automatically generated with make update-readme -->"
	tail="<!-- END of a block automatically generated with make update-readme -->"
	sed -i -e "/^$${lead}/,/^$${tail}/{ /^$${lead}/{p; r assets/glonet_sample.md
	}; /^$${tail}/p; d }" README.md
	rm assets/glonet_sample.md

evaluate: SELECTED_ENVIRONMENT_NAME = ${TEST_ENVIRONMENT_NAME}
evaluate:
	${ACTIVATE_ENVIRONMENT}
	jupyter nbconvert --execute --to notebook $(NOTEBOOK_PATH) --output $(OUTPUT_NAME)

run-tests: SELECTED_ENVIRONMENT_NAME = ${TEST_ENVIRONMENT_NAME}
run-tests:
	${ACTIVATE_ENVIRONMENT}
	pip install --editable .
	$(MAKE) evaluate NOTEBOOK_PATH=assets/glonet_sample.ipynb OUTPUT_NAME=glonet_sample.report.ipynb
	python tests/compare_notebook.py tests/assets/glonet_sample.report.ipynb assets/glonet_sample.report.ipynb
