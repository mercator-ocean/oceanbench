.. SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
..
.. SPDX-License-Identifier: EUPL-1.2

Observation support and provenance
==================================

The optional observation-support report describes the evidence available for
temperature and salinity verification. It accompanies the existing Class IV
scores; it does not change their calculation or the official score table.
Start with a small region and a short forecast period because the report reads
observation audit fields and interpolates the forecast at observation locations.

Create a report
---------------

``assets/observation_support_example.py`` contains a complete, small example
using two days of public GLONET forecasts in the North Atlantic. Its sections
can be run as notebook cells. No model training is required.

.. code-block:: python

   from oceanbench.diagnostics import observation_support, plot_observation_coverage

   report = observation_support(challenger_dataset, region="ibi")
   report.counts
   report.monthly_coverage
   report.quality_control
   report.provenance
   report.notes

   figure = plot_observation_coverage(
       report, variable="sea_water_salinity", depth_bin="5-100m"
   )

The report also accepts ``observations_dataset=...`` for offline analysis. That
dataset must contain observation time, position, depth and scored variables,
with optional audit fields. It may contain raw point records (which the report
pairs with forecast windows) or the reader's already paired records including
``first_day_datetime``. Unknown audit information
must remain unknown; a score-only observation dataset cannot recover discarded
sensor identities or rejection reasons.

Interpret the report
--------------------

* Lead-day counts describe forecast-observation pairs. One measurement may
  verify several overlapping forecasts. They are not independent sample sizes.
* Geographic and monthly coverage deduplicate repeated uses of observations.
  Where native profile identifiers are unavailable, platform and observation
  time identify estimated profile groups. Surface measurements can also form
  such groups; they are not necessarily vertical Argo profiles.
* Geographic bins describe the requested domain and evaluation period. They
  are not an ocean mask, an estimate of the fraction of the ocean observed, or
  an uncertainty map. Empty bins can include land.
* ``minimum_profiles`` is a descriptive sparsity threshold, not a statistical
  significance or confidence threshold. Missing instrument identity limits
  the profile counts and must not be interpreted as absence of measurements.
* Observation availability and forecast availability are different. A valid
  observation without a matching model value does not establish a gap in the
  observing system.
* Quality controls apply to individual variables as well as whole rows. A row
  can retain temperature while its salinity is rejected. The stored
  ``qc_reason`` records the first row-level failure, not every failed test.
  QC totals describe raw records throughout the selected domain and forecast
  windows, including depths outside the scored bins; they need not sum to the
  per-bin score counts. Known current/altimetry-only rows are excluded from
  temperature/salinity QC totals.
* ``evaluated_dates`` and monthly ``evaluated_days`` identify the dates actually
  requested. A month without a forecast window is ``not_evaluated``; a partial
  month is not a survey of the whole month.

Source products and quality controls
-------------------------------------

The existing 2024 observation store draws temperature/salinity profiles and
surface drifter temperatures from the Copernicus Marine
`in situ temperature and salinity product
<https://data.marine.copernicus.eu/product/INSITU_GLO_PHY_TSASSIM_DISCRETE_NRT_013_047/description>`_.
The same store also includes the
`drifter-current product
<https://data.marine.copernicus.eu/product/INSITU_GLO_PHY_UVASSIM_DISCRETE_NRT_013_054/description>`_
and `along-track sea-level anomalies
<https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L3_MY_008_062/description>`_,
which are outside this companion report's variable scope.

The report preserves the policy, source products/files, basis version and
builder hash actually recorded for each daily store. It does not substitute
the current builder's defaults for the policy used to produce an older store.
Two stores can declare the same basis version while recording different
policies, so the version alone does not establish identical processing.

The builder's scored temperature and salinity fields use the source ``TEMP``
and ``PSAL`` values and their quality flags. Optional adjusted fields are
retained for auditing; their presence does not mean the scorer used them.
This report does not reprocess raw Argo files or change the existing observation
basis. Argo's `profile-file guidance
<https://argo.ucsd.edu/data/how-to-use-argo-files/>`_ distinguishes real-time
and adjusted variables and explains their quality flags. Consult the source
product's documentation before changing that basis.

Independence and limits
-----------------------

The generic report leaves observation independence **unknown** because it does
not inspect model training or forecast initialization records. A measurement is not
automatically independent merely because it is stored separately from a
forecast. Establishing independence requires the model's training period and
data lineage, its initialization and assimilation windows, and evidence that
verification observations were unavailable or withheld from those inputs.

For the GLONET example, public evidence supports a more specific assessment:

* The `GLONET paper <https://doi.org/10.1029/2025JH000686>`_ reports training
  on GLORYS12 from 1993 through 2019 and validation in 2020. The January 2024
  evaluation observations are outside those published periods; the forecast
  store does not identify the exact checkpoint that produced it.
* OceanBench identifies GLO12 as GLONET's initial-condition source. The example
  forecast store records ``forecast_reference_time=2024-01-02`` and its first
  forecast day is January 3. Neither field establishes the observation cutoff
  of the initialization snapshot.
* `GLO12's quality document
  <https://catalogue.marine.copernicus.eu/documents/QUID/CMEMS-GLO-QUID-001-024.pdf>`_
  lists the 013_047 temperature/salinity observation product family among its
  assimilated inputs. Sharing a product family does not establish overlap of
  individual measurements.
* The `GLO12 product manual
  <https://documentation.marine.copernicus.eu/PUM/CMEMS-GLO-PUM-001-024.pdf>`_
  explains that recent historical fields are overwritten with later analyses.
  A retrospective download is therefore insufficient to identify the original
  nowcast. The current public inference code is not a manifest for the archived
  benchmark run.

Accordingly, the example is outside the published training/validation periods,
while independence from initialization assimilation remains unverified.
A verified latest-observation cutoff before the evaluation measurements could
establish their temporal exclusion without a complete assimilation ledger.
Historical observations from the same platform do not alone demonstrate leakage
of a later measurement. Record the checkpoint and initialization snapshot,
production time and cutoff before promoting this assessment to verified.

Coverage is conditional on the observations in these source stores, the
requested dates/domain, quality controls and depth bins. It is not an inventory
of every ocean observing system. The temperature and salinity depth bins follow
the existing Class IV implementation, including its endpoint depth clamping
when a forecast does not extend to an observation's depth.

This implementation follows the Class IV selection on the current main branch.
The proposed shared-population changes in
`PR #329 <https://github.com/mercator-ocean/oceanbench/pull/329>`_ are separate;
its eligibility and missing-value rules must be integrated before claiming
agreement with scores produced by that version. No shared-mask rule is copied
into this report.
