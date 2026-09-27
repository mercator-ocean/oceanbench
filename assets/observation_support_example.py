# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime

from IPython.display import display

from oceanbench.datasets.challenger import glonet
from oceanbench.diagnostics import observation_support, plot_observation_coverage
from oceanbench.regions import custom, subset

# # Observation support: a two-day North Atlantic example
# This example uses public GLONET forecasts beginning on January 3, 2024,
# and the observation stores for January 3 and 4. It evaluates temperature
# and salinity in a small region. It does not establish observation independence.

region = custom(
    identifier="north_atlantic_sample",
    display_name="North Atlantic sample",
    minimum_latitude=30,
    maximum_latitude=45,
    minimum_longitude=-40,
    maximum_longitude=-20,
)
source_forecast = glonet([datetime(2024, 1, 3)])
challenger_dataset = subset(source_forecast, region)[["thetao", "so"]].isel(lead_day_index=slice(0, 2))

# ## Available observations and matched forecast values
# Lead days use zero-based indices. Counts refer to forecast-observation pairs,
# not independent samples. The official score table is not modified.

try:
    report = observation_support(challenger_dataset, region=region, spatial_bin_degrees=5)
finally:
    source_forecast.close()
display(report.counts)

# ## Coverage during the evaluated days
# Profile groups are estimates when only platform and observation time are
# available. Geographic bins are not an ocean mask; empty bins may include land.

display(report.monthly_coverage)
display(plot_observation_coverage(report, variable="sea_water_salinity", depth_bin="5-100m"))

# ## Quality controls and recorded sources
# The stored policy is reported separately for each day. A QC rejection reason
# is the first row-level failure, not a complete list of failed checks.

display(report.quality_control)
display(report.provenance)
display(report.notes)
# ## Model-specific public evidence
# https://doi.org/10.1029/2025JH000686 reports training in 1993-2019
# and validation in 2020. See docs/observation-support.rst for initialization
# evidence and the remaining archived-run provenance requirements.

display(
    {
        "published_training_period": "1993-2019",
        "published_validation_period": "2020",
        "evaluation_outside_published_periods": True,
        "forecast_reference_time": source_forecast.attrs.get("forecast_reference_time"),
        "exact_checkpoint_verified": False,
        "initialization_observation_cutoff_verified": False,
        "observation_independence": report.independence_status,
    }
)
