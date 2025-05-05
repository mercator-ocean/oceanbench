<!--
SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# Report an issue or propose an evolution

Feel free to open an issue to discuss a particular change or evolution; however, please be advised that we may not respond to your request or may provide a negative response.

# Contribute

All submissions, including submissions by project members, require review.

## REUSE and licensing

This repository follows the [REUSE software](https://reuse.software) recommendations.
By contributing, you accept that Mercator Ocean International is the copyright holder of all files, and that files are published under the [EUPL-1.2](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12) license.

## Developer Certificate of Origin and sign off agreement

This repository uses the [Developer Certificate of Origin from the Linux Foundation](https://developercertificate.org/).
Commit signoff is required.
Before signing off on a commit, you should ensure that your commit is in compliance with the rules and licensing governing the repository.
For commits made via the Git command line interface, you must sign off on the commit using the `--signoff` option. For more information, see the [Git documentation](https://git-scm.com/docs/git-commit).

## Git workflow

This repository relies on a git workflow using rebase on the `main` branch.
Please rebase your pull requests on `main` and squash your commit into a single one.

More info:

- [A Git Workflow Using Rebase](https://medium.com/singlestone/a-git-workflow-using-rebase-1b1210de83e5)
- [Merging vs. rebasing](https://www.atlassian.com/git/tutorials/merging-vs-rebasing)
- [Official Git rebase documentation](https://git-scm.com/book/en/v2/Git-Branching-Rebasing)

## Reproducibility and automation

We use Micromamba virtual environment manager.
We aim at automating as much as possible everything.
Checkout the [Makefile](Makefile) and CI/Actions to discover what is done automatically and what you can do manually in that project.

## Library life cycle

The library source code is under the `oceanbench` directory:

```sh
└── oceanbench
    ├── core
    │   ├── ...
    │   ├── derived_quantities.py # internal entrypoint
    │   ├── metrics.py # internal entrypoint
    │   └── plot.py # internal entrypoint
    ├── derived_quantities.py # python lib interface
    ├── evaluate.py # python lib interface
    ├── metrics.py # python lib interface
    └── plot.py # python lib interface
```

The main objective of the library is to be use as part of a notebook that execution is used as an evaluation reports.
On new version release, the participating models are re-evaluated.
The OceanBench maintainers stored the versioned evaluation reports of all participating models on [EDITO](https://datalab.dive.edito.eu/my-files/project-oceanbench/public/evaluation-reports/).

### Version management

We are using semantic versioning X.Y.Z → MAJOR.MINOR.PATCH → for example 1.0.2. We follow the SEMVER principles:

>Given a version number MAJOR.MINOR.PATCH, increment the:
>
>- MAJOR version when you make incompatible API changes
>- MINOR version when you add functionality in a backward compatible manner
>- PATCH version when you make backward compatible bug fixes
>
>Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

## Website life cycle

The website source code is under the `website` directory.
The `main` branch is automatically deployed on OceanBench website.
The website parse and display a given version of the evaluation reports stored on [EDITO](https://datalab.dive.edito.eu/my-files/project-oceanbench/public/evaluation-reports/).

## Development environment

Create a conda environment:

```sh
make create-environment
```

After any implementation:

- add test/ documentation on new functionality if relevant

- add necessary module to "pyproject.toml" in [tool.poetry.dependencies] section

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
conda activate copernicusmarine_test
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
