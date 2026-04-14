# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
import shutil

from numcodecs import Blosc

from oceanbench.demo import glonet

COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)


def _write_dataset(dataset, target_path, dtype: str | None = None) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        shutil.rmtree(target_path)
    encoding = {
        variable_name: {
            "compressor": COMPRESSOR,
            **({"dtype": dtype} if dtype else {}),
        }
        for variable_name in dataset.data_vars
    }
    dataset.to_zarr(target_path, mode="w", consolidated=True, encoding=encoding)


def main() -> None:
    print(f"Fetching GLONET demo data into {glonet.demo_data_dir()}")

    print("Opening remote GLONET challenger sample...")
    challenger_dataset = glonet.load_remote_challenger_dataset()
    print("Saving local 1-degree challenger sample...")
    _write_dataset(challenger_dataset, glonet.challenger_path(), dtype="float32")

    print("Opening remote GLORYS 1-degree reference sample...")
    glorys_dataset = glonet.load_remote_reference_dataset(glonet.remote_glorys_path())
    print("Saving local GLORYS reference sample...")
    _write_dataset(glorys_dataset, glonet.glorys_path())

    print("Opening remote GLO12 1-degree reference sample...")
    glo12_dataset = glonet.load_remote_reference_dataset(glonet.remote_glo12_path())
    print("Saving local GLO12 reference sample...")
    _write_dataset(glo12_dataset, glonet.glo12_path())

    print("Opening remote native GLONET challenger SSH sample for eddies...")
    eddy_challenger_dataset = glonet.load_remote_native_challenger_dataset()[glonet.EDDY_SAMPLE_VARIABLES]
    print("Saving local quarter-degree challenger SSH sample...")
    _write_dataset(eddy_challenger_dataset, glonet.eddy_challenger_path(), dtype="float32")

    print("Opening remote native GLORYS SSH sample for eddies...")
    eddy_glorys_dataset = glonet.load_remote_reference_dataset(glonet.remote_glorys_quarter_degree_path())[
        glonet.EDDY_SAMPLE_VARIABLES
    ]
    print("Saving local quarter-degree GLORYS SSH sample...")
    _write_dataset(eddy_glorys_dataset, glonet.eddy_glorys_path(), dtype="float32")

    print("Opening remote native GLO12 SSH sample for eddies...")
    eddy_glo12_dataset = glonet.load_remote_reference_dataset(glonet.remote_glo12_quarter_degree_path())[
        glonet.EDDY_SAMPLE_VARIABLES
    ]
    print("Saving local quarter-degree GLO12 SSH sample...")
    _write_dataset(eddy_glo12_dataset, glonet.eddy_glo12_path(), dtype="float32")

    print("Opening remote Class-4 observation samples...")
    for day_key in glonet.observation_day_keys():
        print(f"Saving local observations for {day_key}...")
        observation_dataset = glonet.load_remote_observation_dataset(day_key)
        _write_dataset(observation_dataset, glonet.observation_path(day_key))

    print("Opening remote mean sea surface height support dataset...")
    mean_sea_surface_height_dataset = glonet.load_remote_mean_sea_surface_height_dataset()
    print("Saving local mean sea surface height support dataset...")
    _write_dataset(mean_sea_surface_height_dataset, glonet.mean_sea_surface_height_path())

    metadata = {
        "sample_first_day": glonet.SAMPLE_FIRST_DAY.strftime("%Y-%m-%d"),
        "challenger_source": glonet.remote_glonet_path(),
        "glorys_source": glonet.remote_glorys_path(),
        "glo12_source": glonet.remote_glo12_path(),
        "glorys_quarter_degree_source": glonet.remote_glorys_quarter_degree_path(),
        "glo12_quarter_degree_source": glonet.remote_glo12_quarter_degree_path(),
        "observation_sources": {
            day_key: glonet.remote_observation_path(day_key) for day_key in glonet.observation_day_keys()
        },
        "mean_sea_surface_height_source": glonet.remote_mean_sea_surface_height_path(),
        "local_challenger_path": str(glonet.challenger_path().relative_to(glonet.project_root())),
        "local_glorys_path": str(glonet.glorys_path().relative_to(glonet.project_root())),
        "local_glo12_path": str(glonet.glo12_path().relative_to(glonet.project_root())),
        "local_eddy_challenger_path": str(glonet.eddy_challenger_path().relative_to(glonet.project_root())),
        "local_eddy_glorys_path": str(glonet.eddy_glorys_path().relative_to(glonet.project_root())),
        "local_eddy_glo12_path": str(glonet.eddy_glo12_path().relative_to(glonet.project_root())),
        "local_observation_paths": [
            str(glonet.observation_path(day_key).relative_to(glonet.project_root()))
            for day_key in glonet.observation_day_keys()
        ],
        "local_mean_sea_surface_height_path": str(
            glonet.mean_sea_surface_height_path().relative_to(glonet.project_root())
        ),
    }
    glonet.metadata_path().parent.mkdir(parents=True, exist_ok=True)
    glonet.metadata_path().write_text(json.dumps(metadata, indent=2), encoding="utf8")

    print("Finished caching the GLONET demo data locally.")


if __name__ == "__main__":
    main()
