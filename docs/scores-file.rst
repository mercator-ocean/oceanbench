.. SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
..
.. SPDX-License-Identifier: EUPL-1.2

.. _scores-file-page:

===============
The scores file
===============

``oceanbench evaluate`` and ``oceanbench.evaluate_challenger`` write one scores file per challenger and region, named ``<challenger>.<region>.scores.json``.
``<challenger>`` is the challenger file name without its ``.py`` extension and ``<region>`` is the region identifier, for example ``glonet.global.scores.json`` or ``glonet.ibi.scores.json``.
The file is written in the current directory, and also uploaded to ``s3://<output-bucket>/<output-prefix>/`` when ``--output-bucket`` is given.

The evaluation also prints every score table and the duration of every step.

Comparable scores
*****************

Two scores files are comparable when they have the same ``oceanbench_version``, the same ``region`` and the same ``start_dates``.
Starting with the next release, the scores of the OceanBench leaderboard are published as scores files, so a challenger evaluated with the same OceanBench version, on the same region and forecast start dates, can be compared with them row by row.
The leaderboard of version 0.6.0 and older versions stays published as report notebooks.
A different OceanBench version can change the metrics or the reference data, see the changelog.

Fields
******

The file is a JSON object with these fields:

``oceanbench_version`` (string)
    The version of the ``oceanbench`` package that computed the scores, for example ``"0.6.0"``.

``challenger`` (string)
    The challenger name, from the challenger file name.

``region`` (object)
    The evaluated region: ``id`` (string), ``display_name`` (string), ``official`` (boolean) and ``bounds``.
    ``bounds`` is ``null`` for an official region.
    For a custom region it is an object with ``minimum_latitude``, ``maximum_latitude``, ``minimum_longitude`` and ``maximum_longitude`` (numbers, in degrees).

``start_dates`` (list of strings)
    The first forecast day of every forecast in the challenger dataset, as ``YYYY-MM-DD``, in the dataset order.

``lead_days_count`` (integer)
    The number of lead days of the challenger dataset.

``challenger_code`` (string)
    The full text of the challenger file that opened the challenger dataset.

``created_at`` (string)
    The time the file was written, in ISO 8601 format with its UTC offset, for example ``"2026-10-01T06:12:08+00:00"``.

``timings`` (object)
    The duration of every step, in seconds: ``open_challenger_dataset``, then one entry per metric function, then ``total``.

``rows`` (list of objects)
    One row per score.

Rows
****

Every row has these fields:

``metric`` (string)
    The metric, from the ``oceanbench.metrics`` function name: ``rmsd_of_variables``, ``rmsd_of_mixed_layer_depth``, ``rmsd_of_geostrophic_currents`` or ``deviation_of_lagrangian_trajectories``.

``reference`` (string)
    What the challenger is compared to: ``glorys_reanalysis``, ``glo12_analysis`` or ``observations``.
    The metric function name is ``<metric>_compared_to_<reference>``.

``variable`` (string)
    The variable name as the score tables print it, in lower case, for example ``temperature``, ``sea surface height``, ``sea level anomaly``, ``mixed layer depth``, ``zonal geostrophic current`` or ``lagrangian trajectory deviation``.

``standard_name`` (string or null)
    The CF standard name of the variable, for example ``sea_water_potential_temperature``.
    ``null`` for the Lagrangian trajectory deviation.
    Class IV sea level anomaly rows reuse ``sea_surface_height_above_geoid``, so use ``variable`` to tell them apart.

``depth`` (string)
    The depth as the score tables print it: ``surface``, ``50m``, ``100m``, ``200m``, ``300m`` or ``500m`` for the gridded scores, a depth bin such as ``0-5m`` or ``300-600m`` or ``15m`` for the Class IV scores, and ``surface`` for the mixed layer depth, the geostrophic currents and the Lagrangian trajectory deviation.

``unit`` (string)
    The unit of ``value``, for example ``m``, ``°C``, ``PSU``, ``m/s`` or ``km``.

``lead_day`` (integer)
    The lead day, starting at 1. The Lagrangian trajectory deviation starts at lead day 2.

``value`` (number or null)
    The score, at full precision. ``null`` when the score could not be computed.

Rows of the gridded scores, compared to ``glorys_reanalysis`` or ``glo12_analysis`` except the Lagrangian trajectory deviation, also have:

``missing_fraction`` (number or null)
    The area-weighted fraction of the scored ocean cells where the reference has a value and the challenger has none, for this variable and depth. It is the same for every lead day.

Rows compared to ``observations`` (Class IV) also have:

``observations`` (integer or null)
    The number of observations of this variable and depth at the first lead day of the Class IV table, the earliest lead day with data for any variable.
    The same count is repeated on every lead-day row of this variable and depth.
    ``null`` when this variable and depth has no observation at that lead day.

``missing`` (integer or null)
    The number of those observations where the challenger has no value, repeated and ``null`` in the same way.

A metric that could not be computed, for example Class IV scores for dates without observations, has no rows.
The printed table gives the reason.

Rows follow the order of the metric functions in ``timings``, then the order of the printed table rows, then the lead days.
To find a score, match on ``metric``, ``reference``, ``variable``, ``depth`` and ``lead_day``: these five fields identify a row.

Example
*******

A shortened scores file of a GLONET sample:

.. code-block:: json

   {
     "oceanbench_version": "0.6.0",
     "challenger": "glonet_sample",
     "region": {"id": "global", "display_name": "Global", "official": true, "bounds": null},
     "start_dates": ["2024-01-03"],
     "lead_days_count": 10,
     "challenger_code": "# Open GLONET forecast sample with xarray\n...",
     "created_at": "2026-10-01T06:12:08+00:00",
     "timings": {
       "open_challenger_dataset": 0.78,
       "rmsd_of_variables_compared_to_glorys_reanalysis": 243.73,
       "total": 2204.86
     },
     "rows": [
       {
         "metric": "rmsd_of_variables",
         "reference": "glorys_reanalysis",
         "variable": "sea surface height",
         "standard_name": "sea_surface_height_above_geoid",
         "depth": "surface",
         "unit": "m",
         "lead_day": 1,
         "value": 0.074807,
         "missing_fraction": 0.000671
       },
       {
         "metric": "rmsd_of_variables",
         "reference": "observations",
         "variable": "temperature",
         "standard_name": "sea_water_potential_temperature",
         "depth": "0-5m",
         "unit": "°C",
         "lead_day": 1,
         "value": 0.613869,
         "observations": 3784,
         "missing": 0
       },
       {
         "metric": "deviation_of_lagrangian_trajectories",
         "reference": "glorys_reanalysis",
         "variable": "lagrangian trajectory deviation",
         "standard_name": null,
         "depth": "surface",
         "unit": "km",
         "lead_day": 2,
         "value": 11.027288
       }
     ]
   }

The scores files of the sample challengers are in the ``assets`` directory of the repository.
They were converted from the 0.6.0 report notebooks, so their values keep 6 decimal places.

Reading a scores file
*********************

With pandas:

.. code-block:: python

   import json

   import pandas

   with open("glonet.global.scores.json", encoding="utf-8") as file:
       scores = json.load(file)

   rows = pandas.DataFrame(scores["rows"])
   rows[(rows["metric"] == "rmsd_of_variables") & (rows["reference"] == "glorys_reanalysis")].pivot_table(
       index=["depth", "variable"], columns="lead_day", values="value", sort=False
   )
