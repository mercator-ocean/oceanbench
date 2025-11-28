# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime, timedelta


def generate_dates(start_date_str, end_date_str, delta_days) -> list[datetime]:
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    return [
        (start_date + timedelta(days=i * delta_days)) for i in range((end_date - start_date).days // delta_days + 1)
    ]
