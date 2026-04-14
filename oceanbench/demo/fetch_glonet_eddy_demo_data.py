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
    print(f"Fetching GLONET eddy demo data into {glonet.eddy_demo_data_dir()}")

    print("Opening remote native GLONET challenger sample...")
    challenger_dataset = glonet.load_remote_native_challenger_dataset()[glonet.EDDY_SAMPLE_VARIABLES]
    print("Saving local quarter-degree challenger SSH sample...")
    _write_dataset(challenger_dataset, glonet.eddy_challenger_path(), dtype="float32")

    print("Opening remote native GLORYS reference sample...")
    glorys_dataset = glonet.load_remote_reference_dataset(glonet.remote_glorys_quarter_degree_path())[
        glonet.EDDY_SAMPLE_VARIABLES
    ]
    print("Saving local quarter-degree GLORYS SSH sample...")
    _write_dataset(glorys_dataset, glonet.eddy_glorys_path(), dtype="float32")

    print("Opening remote native GLO12 reference sample...")
    glo12_dataset = glonet.load_remote_reference_dataset(glonet.remote_glo12_quarter_degree_path())[
        glonet.EDDY_SAMPLE_VARIABLES
    ]
    print("Saving local quarter-degree GLO12 SSH sample...")
    _write_dataset(glo12_dataset, glonet.eddy_glo12_path(), dtype="float32")

    metadata = {
        "sample_first_day": glonet.SAMPLE_FIRST_DAY.strftime("%Y-%m-%d"),
        "variables": glonet.EDDY_SAMPLE_VARIABLES,
        "challenger_source": glonet.remote_glonet_path(),
        "glorys_source": glonet.remote_glorys_quarter_degree_path(),
        "glo12_source": glonet.remote_glo12_quarter_degree_path(),
        "local_challenger_path": str(glonet.eddy_challenger_path().relative_to(glonet.project_root())),
        "local_glorys_path": str(glonet.eddy_glorys_path().relative_to(glonet.project_root())),
        "local_glo12_path": str(glonet.eddy_glo12_path().relative_to(glonet.project_root())),
    }
    glonet.eddy_metadata_path().parent.mkdir(parents=True, exist_ok=True)
    glonet.eddy_metadata_path().write_text(json.dumps(metadata, indent=2), encoding="utf8")

    print("Finished caching the GLONET eddy demo data locally.")


if __name__ == "__main__":
    main()
