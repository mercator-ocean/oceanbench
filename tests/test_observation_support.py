# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as pyplot
import numpy
import pandas
import pytest
import xarray

from oceanbench.core.classIV import rmsd_class4_validation
from oceanbench.core.dataset_utils import Variable
from oceanbench.core.references import observations as observation_reference
from oceanbench.diagnostics import observation_support, plot_observation_coverage


TEMPERATURE = Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()
SALINITY = Variable.SEA_WATER_SALINITY.key()


def _challenger(first_days=("2024-01-01",), lead_days=(0, 1, 2), values=None) -> xarray.Dataset:
    coordinates = {
        "first_day_datetime": numpy.array(first_days, dtype="datetime64[ns]"),
        "lead_day_index": list(lead_days),
        "depth": [0.0, 10.0, 600.0],
        "latitude": [-20.0, 20.0],
        "longitude": [-20.0, 20.0],
    }
    shape = tuple(len(coordinate) for coordinate in coordinates.values())
    model_values = numpy.full(shape, 3.0) if values is None else values
    return xarray.Dataset(
        {
            TEMPERATURE: (tuple(coordinates), model_values),
            SALINITY: (tuple(coordinates), numpy.full(shape, 35.0)),
        },
        coords=coordinates,
        attrs={"model": "synthetic forecast; no declared assimilation history"},
    )


def _observations(times, depths=None, temperature=None, salinity=None, first_days=None, **metadata) -> xarray.Dataset:
    size = len(times)
    variables = {
        "time": ("observations", pandas.to_datetime(times, format="mixed").values),
        "latitude": ("observations", [0.5] * size),
        "longitude": ("observations", [0.5] * size),
        "depth": ("observations", [0.0] * size if depths is None else depths),
        TEMPERATURE: ("observations", [1.0] * size if temperature is None else temperature),
        SALINITY: ("observations", [35.0] * size if salinity is None else salinity),
    }
    if first_days is not None:
        variables["first_day_datetime"] = ("observations", numpy.array(first_days, dtype="datetime64[ns]"))
    variables.update({name: ("observations", values) for name, values in metadata.items()})
    return xarray.Dataset(variables)


def _counts(report, variable=TEMPERATURE, depth_bin="surface") -> pandas.DataFrame:
    return report.counts.loc[
        (report.counts["variable"] == variable) & (report.counts["depth_bin"] == depth_bin)
    ].set_index("lead_day")


def _coverage(report, variable=TEMPERATURE, depth_bin="surface") -> pandas.DataFrame:
    return report.monthly_coverage.loc[
        (report.monthly_coverage["variable"] == variable) & (report.monthly_coverage["depth_bin"] == depth_bin)
    ].set_index("month")


def test_per_lead_counts_and_rmsd_agree_with_class_iv_and_inputs_are_not_mutated() -> None:
    challenger = _challenger()
    challenger[TEMPERATURE].values[:, 1, :, :, :] = numpy.nan
    observations = _observations(
        ["2024-01-01", "2024-01-01T12:00:00", "2024-01-02", "2024-01-03"],
        temperature=[1.0, 2.0, 4.0, 3.0],
        first_days=["2024-01-01"] * 4,
        obs_id=["a", "b", "c", "d"],
        platform_code=["float-a", "float-b", "float-a", "float-a"],
    )
    original_challenger = challenger.copy(deep=True)
    original_observations = observations.copy(deep=True)

    report = observation_support(challenger, observations_dataset=observations)
    counts = _counts(report)
    scores = rmsd_class4_validation(challenger, observations, [Variable.SEA_WATER_POTENTIAL_TEMPERATURE])

    assert counts["available_observations"].to_dict() == {0: 2, 1: 1, 2: 1}
    assert counts["model_matched_observations"].to_dict() == {0: 2, 1: 0, 2: 1}
    assert counts["missing_forecasts"].to_dict() == {0: 0, 1: 1, 2: 0}
    assert scores["Observations"].tolist() == [2]
    assert counts.loc[0, "rmsd"] == pytest.approx(scores.iloc[0]["Lead day 1"])
    assert counts.loc[2, "rmsd"] == pytest.approx(scores.iloc[0]["Lead day 3"])
    assert numpy.isnan(counts.loc[1, "rmsd"])
    xarray.testing.assert_identical(challenger, original_challenger)
    xarray.testing.assert_identical(observations, original_observations)


def test_noncontiguous_model_leads_are_rejected_instead_of_silently_miscounted() -> None:
    with pytest.raises(ValueError, match="lead"):
        observation_support(
            _challenger(lead_days=(0, 2)),
            observations_dataset=_observations(["2024-01-02"], obs_id=["a"], platform_code=["float-a"]),
        )


def test_empty_observations_keep_zero_counts_and_unobserved_coverage() -> None:
    report = observation_support(_challenger(), observations_dataset=_observations([]))

    assert not report.counts.empty
    assert report.counts["available_observations"].eq(0).all()
    assert report.counts["model_matched_observations"].eq(0).all()
    assert report.counts["missing_forecasts"].eq(0).all()
    assert report.counts["rmsd"].isna().all()
    assert report.monthly_coverage["status"].eq("unobserved").all()
    assert report.spatial_coverage["status"].eq("unobserved").all()


def test_quality_control_counts_variable_rejection_inside_a_kept_row() -> None:
    report = observation_support(
        _challenger(),
        observations_dataset=_observations(
            ["2024-01-01"] * 3,
            temperature=[1.0, 2.0, numpy.nan],
            salinity=[numpy.nan, 35.0, numpy.nan],
            obs_id=["a", "b", "c"],
            platform_code=["float-a"] * 3,
            temp_raw=[1.0, 2.0, 3.0],
            psal_raw=[34.0, 35.0, numpy.nan],
            temp_qc=[1, 1, 1],
            psal_qc=[4, 1, 9],
            qc_keep=[1, 1, 0],
            qc_reason=["", "", "position_qc"],
        ),
    )

    quality_control = report.quality_control.set_index("variable")
    temperature = quality_control.loc[TEMPERATURE]
    salinity = quality_control.loc[SALINITY]
    assert temperature["raw_measurements"] == 3
    assert temperature["accepted_measurements"] == 2
    assert temperature["rejected_measurements"] == 1
    assert salinity["raw_measurements"] == 2
    assert salinity["accepted_measurements"] == 1
    assert salinity["rejected_measurements"] == 1
    assert salinity["raw_missing"] == 1
    assert salinity["qc_metadata_available"]
    assert salinity["source_qc_flags"].get("4", salinity["source_qc_flags"].get(4)) == 1
    assert temperature["row_first_failure_reasons"]["position_qc"] == 1


def test_overlap_adds_forecast_pairs_without_multiplying_profile_coverage() -> None:
    report = observation_support(
        _challenger(first_days=("2024-01-01", "2024-01-02"), lead_days=(0, 1)),
        observations_dataset=_observations(
            ["2024-01-02", "2024-01-02"],
            first_days=["2024-01-01", "2024-01-02"],
            obs_id=["same-source-record"] * 2,
            profile_id=["profile-a"] * 2,
            platform_code=["float-a"] * 2,
        ),
        minimum_profiles=2,
    )

    assert _counts(report)["available_observations"].to_dict() == {0: 1, 1: 1}
    coverage = _coverage(report).loc["2024-01"]
    assert coverage["unique_profiles"] == 1
    assert coverage["unique_platforms"] == 1
    assert coverage["status"] == "sparse"
    assert report.independence_status == "unknown"


def test_unknown_identity_is_explicit_and_never_an_invented_profile() -> None:
    report = observation_support(
        _challenger(first_days=("2024-01-01", "2024-01-02"), lead_days=(0, 1)),
        observations_dataset=_observations(["2024-01-02"]),
    )

    coverage = _coverage(report).loc["2024-01"]
    assert coverage["unique_profiles"] == 0
    assert coverage["unique_platforms"] == 0
    assert coverage["unknown_platform_observations"] == 1
    assert coverage["status"] == "identity_unknown"
    assert report.independence_status == "unknown"
    assert "estimated" in " ".join(report.notes).lower()


def test_half_open_depth_bins_keep_temperature_surface_and_exclude_unbinned_depths() -> None:
    depths = [-1.0, 0.0, 0.999, 1.0, 4.999, 5.0, 99.999, 100.0, 299.999, 300.0, 599.999, 600.0, numpy.nan]
    report = observation_support(
        _challenger(lead_days=(0,)),
        observations_dataset=_observations(
            ["2024-01-01"] * len(depths),
            depths=depths,
            obs_id=[f"record-{index}" for index in range(len(depths))],
            platform_code=["float-a"] * len(depths),
        ),
    )

    temperature_counts = report.counts.loc[report.counts["variable"] == TEMPERATURE].set_index("depth_bin")
    salinity_counts = report.counts.loc[report.counts["variable"] == SALINITY].set_index("depth_bin")
    assert temperature_counts["available_observations"].to_dict() == {
        "surface": 3,
        "0-5m": 2,
        "5-100m": 2,
        "100-300m": 2,
        "300-600m": 2,
    }
    assert salinity_counts["available_observations"].to_dict() == {
        "0-5m": 4,
        "5-100m": 2,
        "100-300m": 2,
        "300-600m": 2,
    }


def test_zero_months_are_retained_across_the_evaluation_span() -> None:
    report = observation_support(
        _challenger(first_days=("2024-01-01", "2024-03-01"), lead_days=(0,)),
        observations_dataset=_observations(["2024-01-01"], profile_id=["a"], platform_code=["float-a"]),
    )

    coverage = _coverage(report)
    assert coverage["unique_profiles"].to_dict() == {"2024-01": 1, "2024-02": 0, "2024-03": 0}
    assert coverage["status"].to_dict() == {"2024-01": "sparse", "2024-02": "not_evaluated", "2024-03": "unobserved"}
    assert coverage["evaluated_days"].to_dict() == {"2024-01": 1, "2024-02": 0, "2024-03": 1}


def test_spatial_coverage_retains_empty_and_sparse_cells_without_filling_them() -> None:
    observations = _observations(
        ["2024-01-01"] * 3,
        obs_id=["a", "b", "c"],
        profile_id=["a", "b", "c"],
        platform_code=["float-a", "float-b", "float-c"],
    ).assign(latitude=("observations", [0.5, 1.5, 12.0]))
    report = observation_support(
        _challenger(lead_days=(0,)), observations_dataset=observations, spatial_bin_degrees=10, minimum_profiles=2
    )

    coverage = report.spatial_coverage.loc[
        (report.spatial_coverage["variable"] == TEMPERATURE) & (report.spatial_coverage["depth_bin"] == "surface")
    ]
    assert len(coverage) == 18 * 36
    assert coverage["unique_profiles"].sum() == 3
    assert coverage["status"].value_counts().to_dict() == {"unobserved": 646, "observed": 1, "sparse": 1}
    assert coverage.loc[coverage["status"] == "unobserved", "unique_profiles"].eq(0).all()
    assert "ocean" in " ".join(report.notes).lower()
    figure = plot_observation_coverage(report, variable=TEMPERATURE, depth_bin="surface")
    assert isinstance(figure, matplotlib.figure.Figure)
    assert figure.axes
    pyplot.close(figure)


def test_distinct_stored_source_policies_are_preserved_without_inferred_independence() -> None:
    provenance = [
        {"observation_day": "2024-01-01", "source": "source-a", "qc_policy": {"accepted_qc_flags": [1]}},
        {"observation_day": "2024-01-02", "source": "source-b", "qc_policy": {"accepted_qc_flags": [1, 2]}},
    ]
    observations = _observations(["2024-01-01"], obs_id=["a"], platform_code=["float-a"])
    observations.attrs["observation_provenance"] = json.dumps(provenance)

    report = observation_support(_challenger(), observations_dataset=observations)

    assert report.provenance.to_dict(orient="records") == provenance
    assert report.independence_status == "unknown"
    assert observations.attrs["observation_provenance"] == json.dumps(provenance)


def test_nonfinite_observations_and_forecasts_do_not_count_as_finite_pairs() -> None:
    challenger = _challenger()
    challenger[TEMPERATURE].values[:, 1, :, :, :] = numpy.inf
    report = observation_support(
        challenger,
        observations_dataset=_observations(
            ["2024-01-01", "2024-01-01", "2024-01-02"],
            temperature=[1.0, numpy.inf, 1.0],
            obs_id=["a", "b", "c"],
            platform_code=["float-a"] * 3,
        ),
    )

    counts = _counts(report)
    assert counts["available_observations"].to_dict() == {0: 1, 1: 1, 2: 0}
    assert counts["model_matched_observations"].to_dict() == {0: 1, 1: 0, 2: 0}
    assert counts["missing_forecasts"].to_dict() == {0: 0, 1: 1, 2: 0}
    assert numpy.isnan(counts.loc[1, "rmsd"])


def test_missing_raw_and_quality_control_metadata_remains_unknown() -> None:
    report = observation_support(
        _challenger(lead_days=(0,)),
        observations_dataset=_observations(["2024-01-01"], obs_id=["a"], platform_code=["float-a"]),
    )

    quality_control = report.quality_control.set_index("variable").loc[TEMPERATURE]
    assert not quality_control["qc_metadata_available"]
    assert quality_control["accepted_measurements"] == 1
    for field in ("raw_measurements", "rejected_measurements", "raw_missing"):
        assert pandas.isna(quality_control[field])


def test_mixed_day_audit_availability_keeps_scores_and_marks_raw_totals_unknown() -> None:
    observations = _observations(
        ["2024-01-01", "2024-01-02"],
        first_days=["2024-01-01"] * 2,
        obs_id=["a", "b"],
        platform_code=["float-a", "float-b"],
        temp_raw=[1.0, numpy.nan],
        temp_qc=[1.0, numpy.nan],
        psal_raw=[35.0, numpy.nan],
        psal_qc=[1.0, numpy.nan],
        qc_keep=[1.0, numpy.nan],
        qc_reason=["", numpy.nan],
        source_day=numpy.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
    )
    scored_variables = ["time", "latitude", "longitude", "depth", TEMPERATURE, SALINITY, "obs_id", "platform_code"]
    observations.attrs["observation_provenance"] = json.dumps(
        [
            {
                "date": "2024-01-01",
                "available_variables": scored_variables
                + ["temp_raw", "temp_qc", "psal_raw", "psal_qc", "qc_keep", "qc_reason"],
            },
            {"date": "2024-01-02", "available_variables": scored_variables},
        ]
    )

    report = observation_support(_challenger(lead_days=(0, 1)), observations_dataset=observations)

    assert _counts(report)["available_observations"].to_dict() == {0: 1, 1: 1}
    quality_control = report.quality_control.set_index("variable").loc[TEMPERATURE]
    assert quality_control["accepted_measurements"] == 2
    assert not quality_control["qc_metadata_available"]
    assert quality_control["audited_measurements"] == 1
    assert quality_control["unknown_metadata_measurements"] == 1
    for field in ("raw_measurements", "rejected_measurements", "raw_missing"):
        assert pandas.isna(quality_control[field])


def test_repeated_reports_use_the_current_observations_and_forecast_values() -> None:
    for index in range(8):
        size = index % 3 + 1
        challenger = _challenger(lead_days=(0,))
        challenger[TEMPERATURE].values[:] = index + 2.0
        observations = _observations(
            ["2024-01-01"] * size,
            temperature=[float(index)] * size,
            obs_id=[f"source-{index}-{row}" for row in range(size)],
            platform_code=[f"float-{index}"] * size,
        )

        report = observation_support(challenger, observations_dataset=observations, spatial_bin_degrees=180)

        counts = _counts(report).loc[0]
        assert counts["available_observations"] == size
        assert counts["model_matched_observations"] == size
        assert counts["rmsd"] == pytest.approx(2.0)


def test_nonmidnight_window_retains_coverage_on_its_final_partial_day() -> None:
    report = observation_support(
        _challenger(first_days=("2024-01-31T12:00:00",), lead_days=(0,)),
        observations_dataset=_observations(
            ["2024-02-01T06:00:00"], obs_id=["a"], profile_id=["a"], platform_code=["float-a"]
        ),
        spatial_bin_degrees=180,
    )

    assert _counts(report).loc[0, "available_observations"] == 1
    assert report.evaluated_dates == ("2024-01-31", "2024-02-01")
    assert _coverage(report)["unique_profiles"].to_dict() == {"2024-01": 0, "2024-02": 1}
    assert _coverage(report)["evaluated_days"].to_dict() == {"2024-01": 1, "2024-02": 1}


def test_quality_control_missing_measurements_excludes_other_observation_streams() -> None:
    report = observation_support(
        _challenger(lead_days=(0,)),
        observations_dataset=_observations(
            ["2024-01-01"] * 4,
            temperature=[1.0, numpy.nan, numpy.nan, numpy.nan],
            salinity=[35.0, numpy.nan, numpy.nan, numpy.nan],
            obs_id=["profile", "sst", "current", "satellite"],
            obs_type=[1, 2, 3, 4],
            temp_raw=[1.0, numpy.nan, numpy.nan, numpy.nan],
            psal_raw=[35.0, numpy.nan, numpy.nan, numpy.nan],
            temp_qc=[1, 9, 9, 9],
            psal_qc=[1, 9, 9, 9],
            qc_keep=[1, 0, 1, 1],
            qc_reason=["", "temp_psal_qc", "", ""],
        ),
        spatial_bin_degrees=180,
    )

    quality_control = report.quality_control.set_index("variable")
    assert quality_control.loc[TEMPERATURE, "raw_measurements"] == 1
    assert quality_control.loc[TEMPERATURE, "raw_missing"] == 1
    assert quality_control.loc[TEMPERATURE, "audited_measurements"] == 2
    assert quality_control.loc[SALINITY, "raw_measurements"] == 1
    assert quality_control.loc[SALINITY, "raw_missing"] == 0
    assert quality_control.loc[SALINITY, "audited_measurements"] == 1


def test_report_closes_its_audit_source_but_preserves_the_callers_dataset(monkeypatch) -> None:
    observations = _observations(["2024-01-01"], obs_id=["a"], platform_code=["float-a"])
    close_calls = []
    observations.set_close(lambda: close_calls.append("closed"))
    monkeypatch.setattr(observation_reference, "observation_audit", lambda _challenger: observations)
    challenger = _challenger(lead_days=(0,))

    observation_support(challenger, observations_dataset=observations, spatial_bin_degrees=180)
    assert close_calls == []
    observation_support(challenger, spatial_bin_degrees=180)
    assert close_calls == ["closed"]
