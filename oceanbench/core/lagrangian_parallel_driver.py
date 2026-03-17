# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pickle
import sys
from pathlib import Path

from oceanbench.core.lagrangian_trajectory import _parallel_deviation_of_lagrangian_trajectories


def _main(input_path: str, output_path: str) -> int:
    with Path(input_path).open("rb") as input_handle:
        payload = pickle.load(input_handle)
    deviations = _parallel_deviation_of_lagrangian_trajectories(
        payload["week_tasks"],
        payload["max_workers"],
    )
    with Path(output_path).open("wb") as output_handle:
        pickle.dump(deviations, output_handle, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m oceanbench.core.lagrangian_parallel_driver INPUT OUTPUT")
    return _main(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    raise SystemExit(main())
