# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json

import numpy
import pandas
import pytest
import xarray

from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.references import observations


def _challenger(first_days=("2024-01-03",), lead_days=2) -> xarray.Dataset:
    return xarray.Dataset(
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): numpy.array(first_days, dtype="datetime64[ns]"),
            Dimension.LEAD_DAY_INDEX.key(): numpy.arange(lead_days),
        }
    )


def _day_dataset(day: str, audit=True) -> xarray.Dataset:
    variables = {variable: ("obs", [1.0, numpy.nan]) for variable in observations._consumed_observation_variable_keys()}
    variables[Dimension.TIME.key()] = (
        "obs",
        pandas.to_datetime([f"{day}T01:00:00", f"{day}T12:00:00"]).values.astype("datetime64[ns]").astype("int64"),
    )
    if audit:
        variables.update(
            {
                "obs_id": ("obs", [f"{day}:a", f"{day}:b"]),
                "obs_type": ("obs", numpy.array([1, 3], dtype="int8")),
                "platform_code": ("obs", ["float", "drifter"]),
                "qc_keep": ("obs", numpy.array([1, 0], dtype="int8")),
                "qc_reason": ("obs", ["", "position_qc"]),
                "temp_raw": ("obs", [12.5, 13.0]),
                "temp_qc": ("obs", numpy.array([1, 4], dtype="int8")),
            }
        )
    return xarray.Dataset(
        variables,
        attrs={
            "obs_basis_version": observations.EXPECTED_OBSERVATIONS_BASIS_VERSION,
            "policy": json.dumps({"day": day, "accepted_qc_flags": [1]}),
            "source_datasets": json.dumps({"temperature": f"dataset-{day}"}),
            "builder_script_sha256": f"sha-{day}",
            "source_files": json.dumps([f"{day}.nc"]),
            "build_timestamp_utc": f"{day}T23:00:00+00:00",
        },
    )


def _local_stores(tmp_path, monkeypatch, datasets) -> None:
    for day, dataset in datasets.items():
        dataset.to_zarr(tmp_path / f"{pandas.Timestamp(day):%Y%m%d}.zarr", mode="w", consolidated=True)
    monkeypatch.setattr(
        observations,
        "observation_path",
        lambda day: str(tmp_path / f"{pandas.Timestamp(day):%Y%m%d}.zarr"),
    )


def test_audit_preserves_daily_provenance_and_lazy_evidence(tmp_path, monkeypatch) -> None:
    days = ["2024-01-03", "2024-01-04"]
    sources = {day: _day_dataset(day) for day in days}
    _local_stores(tmp_path, monkeypatch, sources)
    monkeypatch.setattr(
        observations,
        "open_or_create_local_stage_dataset",
        lambda *_, **__: pytest.fail("Audit reader must bypass score staging"),
    )

    with observations.observation_audit(_challenger()) as selected:
        provenance = json.loads(selected.attrs["observation_provenance"])
        assert selected.sizes == {"observations": 4}
        assert selected["temp_raw"].chunks is not None
        assert selected["obs_id"].values.tolist() == [f"{day}:{identity}" for day in days for identity in ["a", "b"]]
        assert selected["qc_keep"].values.tolist() == [1, 0, 1, 0]
        assert selected["temp_raw"].values.tolist() == [12.5, 13.0, 12.5, 13.0]
        assert selected["temp_qc"].values.tolist() == [1, 4, 1, 4]
        assert numpy.issubdtype(selected[Dimension.TIME.key()].dtype, numpy.datetime64)
        for entry, day in zip(provenance, days):
            assert entry["date"] == day
            assert entry["source_url"] == observations.observation_path(numpy.datetime64(day))
            for key, value in sources[day].attrs.items():
                assert entry[key] == value
            assert "qc_keep" in entry["available_variables"]
        assert "policy" not in selected.attrs
        for key in observations._consumed_observation_variable_keys():
            assert selected[key].attrs["standard_name"] == key


def test_audit_preserves_identity_for_overlapping_windows(tmp_path, monkeypatch) -> None:
    days = ["2024-01-03", "2024-01-04", "2024-01-05"]
    _local_stores(tmp_path, monkeypatch, {day: _day_dataset(day) for day in days})

    with observations.observation_audit(_challenger(first_days=days[:2])) as selected:
        assert selected["obs_id"].values.tolist() == [
            "2024-01-03:a",
            "2024-01-03:b",
            "2024-01-04:a",
            "2024-01-04:b",
            "2024-01-04:a",
            "2024-01-04:b",
            "2024-01-05:a",
            "2024-01-05:b",
        ]
        assert (
            selected[Dimension.FIRST_DAY_DATETIME.key()].values.tolist()
            == numpy.repeat(numpy.array(days[:2], dtype="datetime64[ns]"), 4).tolist()
        )
        assert (
            selected["source_day"].values.astype("datetime64[D]").tolist()
            == numpy.array(
                [days[0], days[0], days[1], days[1], days[1], days[1], days[2], days[2]], dtype="datetime64[D]"
            ).tolist()
        )
        assert len(json.loads(selected.attrs["observation_provenance"])) == 3


def test_audit_missing_fields_remain_unknown(tmp_path, monkeypatch) -> None:
    _local_stores(tmp_path, monkeypatch, {"2024-01-03": _day_dataset("2024-01-03", audit=False)})

    with observations.observation_audit(_challenger(lead_days=1)) as selected:
        assert "qc_keep" not in selected
        assert "qc_reason" not in selected
        assert "obs_id" not in selected
        assert "qc_keep" not in json.loads(selected.attrs["observation_provenance"])[0]["available_variables"]


def test_audit_partial_missing_fields_are_null_not_zero(tmp_path, monkeypatch) -> None:
    _local_stores(
        tmp_path,
        monkeypatch,
        {"2024-01-03": _day_dataset("2024-01-03"), "2024-01-04": _day_dataset("2024-01-04", audit=False)},
    )

    with observations.observation_audit(_challenger()) as selected:
        assert selected["qc_keep"].values[:2].tolist() == [1, 0]
        assert numpy.isnan(selected["qc_keep"].values[2:]).all()
        assert pandas.isna(selected["obs_id"].values[2:]).all()
        assert "qc_keep" not in json.loads(selected.attrs["observation_provenance"])[1]["available_variables"]


def test_scoring_reader_still_drops_audit_fields(tmp_path, monkeypatch) -> None:
    days = ["2024-01-03", "2024-01-04"]
    _local_stores(tmp_path, monkeypatch, {day: _day_dataset(day) for day in days})
    monkeypatch.setattr(observations, "_should_stage_observations_locally", lambda: False)

    with observations.observations(_challenger()) as selected:
        assert "obs_id" not in selected
        assert "temp_raw" not in selected
        assert "qc_keep" not in selected
        assert "observation_provenance" not in selected.attrs
        assert selected.sizes == {"observations": 4}


def test_audit_requires_basis_on_every_day(tmp_path, monkeypatch) -> None:
    second_day = _day_dataset("2024-01-04")
    second_day.attrs["obs_basis_version"] = "2024-v2.0.1"
    _local_stores(tmp_path, monkeypatch, {"2024-01-03": _day_dataset("2024-01-03"), "2024-01-04": second_day})

    with pytest.raises(observations.ObservationBasisVersionError, match="2024-v2.0.1"):
        observations.observation_audit(_challenger())


def test_audit_missing_day_raises_and_closes_opened_sources(monkeypatch) -> None:
    source = _day_dataset("2024-01-03")
    closed = []
    source.set_close(lambda: closed.append(True))

    def open_day(url, **_):
        if url.endswith("20240103.zarr"):
            return source
        raise FileNotFoundError("Missing observation day")

    monkeypatch.setattr(observations, "open_remote_zarr", open_day)
    with pytest.raises(FileNotFoundError, match="Missing observation day"):
        observations.observation_audit(_challenger())
    assert closed == [True]


def test_audit_closes_owned_sources_after_use(monkeypatch) -> None:
    source = _day_dataset("2024-01-03")
    closed = []
    source.set_close(lambda: closed.append(True))
    monkeypatch.setattr(observations, "open_remote_zarr", lambda *_, **__: source)

    selected = observations.observation_audit(_challenger(lead_days=1))
    assert closed == []
    selected.close()
    assert closed == [True]


def test_audit_opens_only_requested_days_and_preserves_reversed_start_order(tmp_path, monkeypatch) -> None:
    days = ["2024-01-03", "2024-03-04"]
    _local_stores(tmp_path, monkeypatch, {day: _day_dataset(day) for day in days})

    with observations.observation_audit(_challenger(first_days=days[::-1], lead_days=1)) as selected:
        assert selected["obs_id"].values.tolist() == [
            f"{day}:{identity}" for day in days[::-1] for identity in ["a", "b"]
        ]
        assert [entry["date"] for entry in json.loads(selected.attrs["observation_provenance"])] == days


def test_audit_metadata_preserves_conflicting_stored_attributes(monkeypatch) -> None:
    source = _day_dataset("2024-01-03")
    source.attrs.update(date="stored-date", source_url="stored-url", available_variables=["stored-variables"])
    monkeypatch.setattr(observations, "open_remote_zarr", lambda *_, **__: source)

    with observations.observation_audit(_challenger(lead_days=1)) as selected:
        provenance = json.loads(selected.attrs["observation_provenance"])[0]
        assert provenance["date"] == "2024-01-03"
        assert provenance["source_url"] == observations.observation_path(numpy.datetime64("2024-01-03"))
        assert provenance["source_attributes"] == source.attrs


@pytest.mark.parametrize("challenger", [_challenger(first_days=()), _challenger(lead_days=0)])
def test_audit_empty_windows_raise_clear_error(challenger) -> None:
    with pytest.raises(ValueError, match="at least one forecast start and one lead day"):
        observations.observation_audit(challenger)


def test_audit_missing_forecast_datetime_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="must contain valid datetimes"):
        observations.observation_audit(_challenger(first_days=("NaT",)))


def test_audit_nonmidnight_window_opens_its_last_partial_day(tmp_path, monkeypatch) -> None:
    days = ["2024-01-03", "2024-01-04"]
    _local_stores(tmp_path, monkeypatch, {day: _day_dataset(day) for day in days})

    with observations.observation_audit(_challenger(first_days=("2024-01-03T12:00:00",), lead_days=1)) as selected:
        assert selected["obs_id"].values.tolist() == ["2024-01-03:b", "2024-01-04:a"]


def test_audit_decodes_each_days_datetime_basis_before_concatenating(tmp_path, monkeypatch) -> None:
    days = ["2024-01-03", "2024-01-04", "2024-01-05"]
    datasets = {day: _day_dataset(day) for day in days}
    for day in days[:2]:
        datasets[day]["time_ns"] = datasets[day]["time"].astype("datetime64[ns]")
    _local_stores(tmp_path, monkeypatch, datasets)

    with observations.observation_audit(_challenger(lead_days=3)) as selected:
        assert numpy.issubdtype(selected["time_ns"].dtype, numpy.datetime64)
        numpy.testing.assert_array_equal(selected["time_ns"].values[:4], selected["time"].values[:4])
        assert numpy.isnat(selected["time_ns"].values[4:]).all()
        assert selected["qc_keep"].values.tolist() == [1, 0, 1, 0, 1, 0]
