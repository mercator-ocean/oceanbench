<!--
SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# Report an issue or propose an evolution

Feel free to open an issue to discuss a particular change or evolution; however, please be advised that we may not be able to respond to your request or may decline it.

# Contribute

All submissions, including submissions by project members, require review.

## REUSE and licensing

This repository follows the [REUSE software](https://reuse.software) recommendations.
By contributing, you acknowledge that Mercator Ocean International is the copyright holder of all files, and that files are published under the [EUPL-1.2](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12) license.

## Developer Certificate of Origin and sign off agreement

This repository uses the [Developer Certificate of Origin from the Linux Foundation](https://developercertificate.org/).
Commit signoff is required.
Before signing off on a commit, you should ensure that your commit is in compliance with the rules and licensing governing the repository.
For commits made via the Git command line interface, you must sign off on the commit using the `--signoff` option. For more information, see the [Git documentation](https://git-scm.com/docs/git-commit).

## Git workflow

This repository relies on a git workflow using rebase on the `main` branch.
Please rebase your branches on `main` and squash your commit into a single one.
Always review the code yourself before opening pull requests, and check they are compliant with the [development guidelines](#development-guidelines-and-conventions).

More info:

- [A Git Workflow Using Rebase](https://medium.com/singlestone/a-git-workflow-using-rebase-1b1210de83e5)
- [Merging vs. rebasing](https://www.atlassian.com/git/tutorials/merging-vs-rebasing)
- [Official Git rebase documentation](https://git-scm.com/book/en/v2/Git-Branching-Rebasing)

## Reproducibility and automation

We use the Micromamba virtual environment manager.
We aim to automate as much as possible.
Check out the [Makefile](Makefile) and CI/Actions to discover what is done automatically and what you can do manually in this project.

## Library life cycle

The main objective of the library is to be used as part of a notebook whose execution serves as an evaluation report.
On each new version release, the participating models are re-evaluated.
The OceanBench maintainers store the versioned evaluation reports of all participating models on [EDITO](https://datalab.dive.edito.eu/my-files/project-oceanbench/public/evaluation-reports/).

The library source code is under the `oceanbench` directory:

```sh
└── oceanbench/
    ├── core
    │   ├── ...
    │   ├── metrics.py
    │   ├── mixed_layer_depth.py
    │   ├── references
    │   │   └── ...
    │   ├── rmsd.py
    │   └── version.py
    ├── cli.py # CLI API
    ├── datasets
    │   ├── ...
    │   ├── challenger.py # Python API to open challenger datasets
    │   ├── input.py # Python API to open input datasets
    │   └── reference.py # Python API to open reference datasets
    ├── __init__.py # Python main API
    └── metrics.py # Python metrics API
```

All core/domain related code is in the `core` subdirectory, while all API/interface related code is outside it, starting at the root for the main CLI and Python APIs.

### Version management

We are using semantic versioning X.Y.Z → MAJOR.MINOR.PATCH → for example 1.0.2. We follow the SEMVER principles:

>Given a version number MAJOR.MINOR.PATCH, increment the:
>
>- MAJOR version when you make incompatible API changes
>- MINOR version when you add functionality in a backward compatible manner
>- PATCH version when you make backward compatible bug fixes
>
>Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

Using this version management, it is possible to publish a new release with one of the following command:
```sh
make release-major
make release-minor
make release-patch
```

## Website life cycle

The website source code is under the `website` directory.
The `main` branch is automatically deployed to the OceanBench website.
The website parses and displays a given version of the evaluation reports stored on [EDITO](https://datalab.dive.edito.eu/my-files/project-oceanbench/public/evaluation-reports/).

## Development guidelines and conventions

The development has no hard-defined code style, but we aims at following some principles, whoever/whatever produces the code:

1. Name files, variables, constants, modules, functions, etc. with complete human-readable words.
Do not use acronyms, abbreviations, or abstract names if it can be avoided.
2. Keep the code functional (like in functional programming) as much as Python allows you. It starts by never mutating variables, writing functions, and splitting big functions into well named smaller functions.
3. Only use docstrings to document API functions.
Other inline comments that restate the code should be useless if you name everything well, and will be outdated soon anyway.
4. No need to abuse of architecture paradigms such as generic classes and heritage.
Just write functions that make sense to the maintainers and for the future developers.
5. For Python code, prefix with `_` the name of the functions only used in the file.
6. For Python code related to metrics computation, delegate everything (as much as possible) to `xarray` (that will distribute computation over `dask` automatically under the hood).
That starts by favoring dataset/matrix computations over using conditions, loops and mappings. If you use `if` and `for` statements for example, that is symptomatic and can have huge performance consequences.

## Development environment

Create a conda environment:

```sh
make create-environment
```

After any implementation:

- add tests/documentation for new functionality if relevant

- add necessary modules to "pyproject.toml" in the [tool.poetry.dependencies] section

- run pre-commit before committing:

``` sh
pre-commit run --all-files
```

or

```sh
make check-format
```

- run tests

## Tests

### Run tests on Linux

Create a test conda environment:

```sh
make create-test-environment
```

Then activate this test environment:

```sh
conda activate oceanbench_test
```

Export credentials to local variables (if you don't use `moi`, simply put your own credentials):

```sh
export COPERNICUSMARINE_SERVICE_USERNAME=$(moi read-secret --name COPERNICUSMARINE_SERVICE_USERNAME)
export COPERNICUSMARINE_SERVICE_PASSWORD=$(moi read-secret --name COPERNICUSMARINE_SERVICE_PASSWORD)
```

Finally, run the tests:

```sh
make run-tests
```
