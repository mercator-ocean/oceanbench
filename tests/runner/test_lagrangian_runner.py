# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pytest
import xarray

from oceanbench.core import lagrangian_trajectory
from oceanbench.runner import records, run


class _StopAfterCount(Exception):
    pass


def test_lagrangian_records_scale_particles_from_the_global_dataset(monkeypatch):
    global_dataset = xarray.Dataset(attrs={"name": "global"})
    regional_dataset = xarray.Dataset(attrs={"name": "regional"})
    seen = {}

    def fake_count(global_challenger, regional_challenger):
        seen["global"] = global_challenger.attrs["name"]
        seen["regional"] = regional_challenger.attrs["name"]
        raise _StopAfterCount

    monkeypatch.setattr(lagrangian_trajectory, "lagrangian_particle_count_for_region", fake_count)
    monkeypatch.setattr(run, "subset_dataset_to_region", lambda dataset, region: dataset)
    context = records.RunContext(
        challenger="glonet",
        challenger_version="0.0.0",
        year=2024,
        region="ibi",
        oceanbench_version="0.0.0",
    )

    with pytest.raises(_StopAfterCount):
        run._lagrangian_records(
            global_dataset,
            regional_dataset,
            reference_name="glorys",
            reference_openers={"glorys": lambda challenger: challenger},
            region="ibi",
            context=context,
        )

    assert seen == {"global": "global", "regional": "regional"}
