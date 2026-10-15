# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
import logging

from oceanbench.core import local_stage


def test_waiting_on_a_stage_lock_warns_once_after_the_wait_threshold(tmp_path, monkeypatch, caplog) -> None:
    stage_path = tmp_path / "reference-glorys-quarter_degree-10d" / "20240103.zarr"
    lock_path = stage_path.with_name(f"{stage_path.name}.lock")
    lock_path.mkdir(parents=True)
    owner = json.dumps({"created_at": "2026-10-15T08:00:00+00:00", "hostname": "lir-node", "pid": 1234})
    (lock_path / "owner.json").write_text(owner, encoding="utf-8")
    clock = [1000.0]
    warnings_seen_at_each_poll = []

    def wait_until_the_lock_owner_finishes(_seconds) -> None:
        warnings_seen_at_each_poll.append(len(caplog.records))
        if len(warnings_seen_at_each_poll) == 1:
            clock[0] += local_stage.LOCAL_STAGE_LOCK_WAIT_WARNING_SECONDS - 1
        elif len(warnings_seen_at_each_poll) == 2:
            clock[0] += 1
        elif len(warnings_seen_at_each_poll) == 3:
            clock[0] += local_stage.LOCAL_STAGE_LOCK_WAIT_WARNING_SECONDS
        else:
            stage_path.mkdir()

    monkeypatch.setattr(local_stage, "monotonic", lambda: clock[0])
    monkeypatch.setattr(local_stage, "sleep", wait_until_the_lock_owner_finishes)
    built = []
    caplog.set_level(logging.WARNING, logger=local_stage.__name__)

    local_stage.ensure_local_stage(stage_path, built.append)

    assert built == []
    assert warnings_seen_at_each_poll == [0, 0, 1, 1]
    assert [record.getMessage() for record in caplog.records] == [
        f"Waiting for the stage lock {lock_path} held by {owner}. If that job was killed, delete the lock directory."
    ]
