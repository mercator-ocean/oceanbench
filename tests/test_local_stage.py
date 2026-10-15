# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
import logging

from oceanbench.core import local_stage


def test_waiting_on_a_stage_lock_logs_its_owner_once(tmp_path, monkeypatch, caplog) -> None:
    stage_path = tmp_path / "reference-glorys-quarter_degree-10d" / "20240103.zarr"
    lock_path = stage_path.with_name(f"{stage_path.name}.lock")
    lock_path.mkdir(parents=True)
    owner = json.dumps({"created_at": "2026-10-15T08:00:00+00:00", "hostname": "lir-node", "pid": 1234})
    (lock_path / "owner.json").write_text(owner, encoding="utf-8")
    polls = []

    def wait_until_the_lock_owner_finishes(_seconds) -> None:
        polls.append(_seconds)
        if len(polls) == 3:
            stage_path.mkdir()

    monkeypatch.setattr(local_stage, "sleep", wait_until_the_lock_owner_finishes)
    built = []
    caplog.set_level(logging.WARNING, logger=local_stage.__name__)

    local_stage.ensure_local_stage(stage_path, built.append)

    assert built == []
    assert len(polls) == 3
    assert [record.getMessage() for record in caplog.records] == [
        f"Waiting for the stage lock {lock_path} held by {owner}. If that job was killed, delete the lock directory."
    ]
