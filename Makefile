# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

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

create-test-environment: SELECTED_ENVIRONMENT_NAME = ${TEST_ENVIRONMENT_NAME}
create-test-environment: SELECTED_ENVIRONMENT_FILE_NAME = ${TEST_ENVIRONMENT_FILE_NAME}
create-test-environment: _create-update-environment

check-format: SELECTED_ENVIRONMENT_NAME = ${ENVIRONMENT_NAME}
check-format:
	${ACTIVATE_ENVIRONMENT}
	pre-commit install
	pre-commit run --all-files --show-diff-on-failure

reuse-annotate: SELECTED_ENVIRONMENT_NAME = ${ENVIRONMENT_NAME}
reuse-annotate:
	${ACTIVATE_ENVIRONMENT}
	reuse annotate --year 2025 --copyright "Mercator Ocean International <https://www.mercator-ocean.eu/>" --license EUPL-1.2 --recursive . --skip-unrecognised
	reuse download --all

_generate-notebook: SELECTED_ENVIRONMENT_NAME = ${TEST_ENVIRONMENT_NAME}
_generate-notebook:
	${ACTIVATE_ENVIRONMENT}
	pip install --editable .
	python -c 'import oceanbench; oceanbench.generate_notebook_to_evaluate("$(PYTHON_FILE_PATH)", "$(NOTEBOOK_NAME)")'

_evaluate-notebook: SELECTED_ENVIRONMENT_NAME = ${TEST_ENVIRONMENT_NAME}
_evaluate-notebook:
	${ACTIVATE_ENVIRONMENT}
	jupyter nbconvert --execute --to notebook $(NOTEBOOK_NAME) --inplace --allow-errors

_upload-notebook: SELECTED_ENVIRONMENT_NAME = ${TEST_ENVIRONMENT_NAME}
_upload-notebook:
	${ACTIVATE_ENVIRONMENT}
	s3cmd put $(NOTEBOOK_NAME) s3://$(OUTPUT_BUCKET)/$(OUTPUT_PREFIX)/$(NOTEBOOK_NAME) --access_key=$(AWS_ACCESS_KEY_ID) --secret_key=$(AWS_SECRET_ACCESS_KEY) --access_token=$(AWS_SESSION_TOKEN) --host=$(AWS_S3_ENDPOINT) --host-bucket=$(AWS_S3_ENDPOINT)

evaluate-challenger: SELECTED_ENVIRONMENT_NAME = ${TEST_ENVIRONMENT_NAME}
evaluate-challenger:
	$(MAKE) _generate-notebook PYTHON_FILE_PATH=$(CHALLENGER_PYTHON_FILE_PATH) NOTEBOOK_NAME=$(CHALLENGER_REPORT_NAME)
	$(MAKE) _evaluate-notebook NOTEBOOK_NAME=$(CHALLENGER_REPORT_NAME)
	version=`python -c 'import oceanbench; print(oceanbench.__version__)'`
	$(MAKE) _upload-notebook NOTEBOOK_NAME=$(CHALLENGER_REPORT_NAME) OUTPUT_BUCKET=project-oceanbench OUTPUT_PREFIX="public/evaluation-reports/$${version}"

run-tests: SELECTED_ENVIRONMENT_NAME = ${TEST_ENVIRONMENT_NAME}
run-tests:
	${ACTIVATE_ENVIRONMENT}
	$(MAKE) evaluate-challenger CHALLENGER_PYTHON_FILE_PATH=assets/glonet_sample.py CHALLENGER_REPORT_NAME=glonet_sample.report.ipynb
	python tests/compare_notebook.py assets/glonet_sample.report.ipynb glonet_sample.report.ipynb

_release: SELECTED_ENVIRONMENT_NAME = ${ENVIRONMENT_NAME}
_release:
	${ACTIVATE_ENVIRONMENT}
	BUMP_TYPE=${BUMP_TYPE} ./release.sh

release-patch: BUMP_TYPE = patch
release-patch: _release

release-minor: BUMP_TYPE = minor
release-minor: _release

release-major: BUMP_TYPE = major
release-major: _release
