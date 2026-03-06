# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2


LEAD_DAY_LABEL_PREFIX = "Lead day "


def lead_day_labels(start: int, stop: int) -> list[str]:
    return list(
        map(
            lambda day_index: f"{LEAD_DAY_LABEL_PREFIX}{day_index}",
            range(start, stop + 1),
        )
    )
