def lead_day_labels(start: int, stop: int) -> list[str]:
    return list(
        map(
            lambda day_index: f"Lead day {day_index}",
            range(start, stop + 1),
        )
    )
