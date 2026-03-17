# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

def lead_day_labels(start: int, stop: int) -> list[str]:
    return [f"Lead day {day_index}" for day_index in range(start, stop + 1)]
