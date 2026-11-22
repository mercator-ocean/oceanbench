# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import shutil
import tempfile
import weakref
from pathlib import Path

import pandas


class Class4Matchups:
    """
    The Class IV match-ups, read one part at a time.

    A part holds the rows of one variable and one lead day, of one start date or of all of them, in the
    order of the match-up table. The match-ups of a year do not fit in memory: ``Class4Matchups.on_disk``
    writes them to a temporary directory as they are computed, one file per variable, start date and lead
    day. ``Class4Matchups.in_memory`` reads the parts of a match-up table held in memory.
    """

    def __init__(self, variable_order: list[str] | None = None) -> None:
        self._variable_order = variable_order
        self._variables: list[str] = []
        self._first_days: set[pandas.Timestamp] = set()
        self._scored_first_days: set[pandas.Timestamp] = set()
        self._row_count = 0

    @staticmethod
    def on_disk(variable_order: list[str] | None = None) -> "Class4Matchups":
        """Match-ups to ``add``, listed in ``variable_order`` when given, else in the order they are added."""
        return _DiskClass4Matchups(variable_order)

    @staticmethod
    def in_memory(matchups: pandas.DataFrame) -> "Class4Matchups":
        return _MemoryClass4Matchups(matchups)

    def __len__(self) -> int:
        return self._row_count

    def variables(self) -> list[str]:
        """The variables, in the order of the match-up table."""
        if self._variable_order is None:
            return list(self._variables)
        return sorted(self._variables, key=self._variable_order.index)

    def first_days(self) -> list[pandas.Timestamp]:
        """The start dates with match-ups, sorted."""
        return sorted(self._first_days)

    def scored_first_days(self) -> list[pandas.Timestamp]:
        """The start dates with match-ups that have both an observation and a challenger value, sorted."""
        return sorted(self._scored_first_days)

    def lead_days(self, variable: str) -> list[int]:
        raise NotImplementedError

    def part(self, variable: str, lead_day: int, first_day: pandas.Timestamp | None = None) -> pandas.DataFrame:
        """The rows of ``variable`` at ``lead_day``, of ``first_day`` or of every start date."""
        raise NotImplementedError

    def start(self, first_day: pandas.Timestamp) -> pandas.DataFrame:
        """
        The rows of ``first_day``. Rows of the same variable and lead day are in the order of the match-up table
        and variables come in the order of ``variables``; rows of different lead days may be reordered.
        """
        raise NotImplementedError

    def to_dataframe(self) -> pandas.DataFrame:
        """All the rows, by variable; rows of the same variable, start date and lead day in table order."""
        raise NotImplementedError

    def _count(self, variable: str, rows: pandas.DataFrame) -> None:
        if variable not in self._variables:
            self._variables.append(variable)
        self._row_count += len(rows)
        self._first_days.update(pandas.Timestamp(value) for value in rows["first_day"].dropna().unique())
        is_scored = rows["observation_value"].notna() & rows["model_value"].notna()
        self._scored_first_days.update(
            pandas.Timestamp(value) for value in rows.loc[is_scored, "first_day"].dropna().unique()
        )


class _MemoryClass4Matchups(Class4Matchups):
    def __init__(self, matchups: pandas.DataFrame) -> None:
        super().__init__()
        self._matchups = matchups
        for variable, rows in matchups.groupby("variable", sort=False):
            self._count(variable, rows)

    def lead_days(self, variable: str) -> list[int]:
        return sorted(self._matchups.loc[self._matchups["variable"] == variable, "lead_day"].unique().tolist())

    def part(self, variable: str, lead_day: int, first_day: pandas.Timestamp | None = None) -> pandas.DataFrame:
        selected = (self._matchups["variable"] == variable) & (self._matchups["lead_day"] == lead_day)
        if first_day is not None:
            selected &= self._matchups["first_day"] == first_day
        return self._matchups.loc[selected]

    def start(self, first_day: pandas.Timestamp) -> pandas.DataFrame:
        return self._matchups.loc[self._matchups["first_day"] == first_day]

    def to_dataframe(self) -> pandas.DataFrame:
        return self._matchups


def _feather():
    # pyarrow comes with the viewer extra, the only user of the match-ups.
    import pyarrow.feather

    return pyarrow.feather


def _remove_directory(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


class _DiskClass4Matchups(Class4Matchups):
    def __init__(self, variable_order: list[str] | None) -> None:
        super().__init__(variable_order)
        self._directory = Path(tempfile.mkdtemp(prefix="oceanbench-class4-matchups-"))
        self._remove = weakref.finalize(self, _remove_directory, str(self._directory))
        self._files: dict[tuple[str, int, pandas.Timestamp], Path] = {}
        self._empty: pandas.DataFrame = pandas.DataFrame()
        self._in_memory: dict[str, _MemoryClass4Matchups] = {}
        self._last_first_days: dict[str, pandas.Timestamp] = {}

    def add(self, variable: str, rows: pandas.DataFrame) -> None:
        """
        Keep match-ups of ``variable``, in the order of the table, after those already kept for it: either all its
        start dates at once, or one start date after the other.
        """
        if not len(rows):
            return
        if not self._variables:
            self._empty = rows.iloc[0:0].reset_index(drop=True)
        self._count(variable, rows)
        first_days = rows["first_day"]
        last_first_day = self._last_first_days.get(variable)
        in_start_order = first_days.is_monotonic_increasing and not first_days.isna().any()
        if in_start_order and last_first_day is not None:
            in_start_order = first_days.iloc[0] > last_first_day
        if variable in self._in_memory or not in_start_order:
            # Reading the start dates one after the other would reorder the rows of a lead day.
            if variable in self._last_first_days:
                raise ValueError(f"Match-ups of {variable} are added out of start date order.")
            previous = [self._in_memory[variable].to_dataframe()] if variable in self._in_memory else []
            self._in_memory[variable] = _MemoryClass4Matchups(pandas.concat([*previous, rows], ignore_index=True))
            return
        self._last_first_days[variable] = first_days.iloc[-1]
        for (first_day, lead_day), part in rows.groupby(["first_day", "lead_day"], sort=False):
            path = self._directory / f"{len(self._files)}.arrow"
            _feather().write_feather(part.reset_index(drop=True), path, compression="zstd")
            self._files[(variable, int(lead_day), pandas.Timestamp(first_day))] = path

    def lead_days(self, variable: str) -> list[int]:
        if variable in self._in_memory:
            return self._in_memory[variable].lead_days(variable)
        return sorted({lead_day for file_variable, lead_day, _ in self._files if file_variable == variable})

    def _read(self, variable: str, lead_day: int, first_days: list[pandas.Timestamp]) -> pandas.DataFrame:
        parts = [
            _feather().read_feather(self._files[(variable, lead_day, first_day)])
            for first_day in first_days
            if (variable, lead_day, first_day) in self._files
        ]
        return pandas.concat(parts, ignore_index=True) if parts else self._empty

    def part(self, variable: str, lead_day: int, first_day: pandas.Timestamp | None = None) -> pandas.DataFrame:
        if variable in self._in_memory:
            return self._in_memory[variable].part(variable, lead_day, first_day).reset_index(drop=True)
        return self._read(variable, lead_day, self.first_days() if first_day is None else [first_day])

    def start(self, first_day: pandas.Timestamp) -> pandas.DataFrame:
        parts = [
            self.part(variable, lead_day, first_day)
            for variable in self._variables
            for lead_day in self.lead_days(variable)
        ]
        parts = [part for part in parts if len(part)]
        return pandas.concat(parts, ignore_index=True) if parts else self._empty

    def to_dataframe(self) -> pandas.DataFrame:
        parts = [
            self.part(variable, lead_day, first_day)
            for variable in self._variables
            for first_day in self.first_days()
            for lead_day in self.lead_days(variable)
        ]
        parts = [part for part in parts if len(part)]
        return pandas.concat(parts, ignore_index=True) if parts else self._empty


def class4_matchups(matchups: "pandas.DataFrame | Class4Matchups") -> Class4Matchups:
    """The match-ups as parts, whether they are a table in memory or already parts."""
    return matchups if isinstance(matchups, Class4Matchups) else Class4Matchups.in_memory(matchups)
