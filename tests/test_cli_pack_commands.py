# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Argument-parsing tests for the pack and viewer subcommands (no data is read)."""

import pytest

from oceanbench.cli import _build_parser


def _parse(*arguments):
    return _build_parser().parse_args(list(arguments))


def test_build_pack_defaults_to_a_full_native_resolution_pack_with_baselines():
    args = _parse("build-pack", "/tmp/pack-full-2024-global")

    assert args.command == "build-pack"
    assert args.output == "/tmp/pack-full-2024-global"
    assert args.kind == "full"
    assert args.resolution is None
    assert args.baselines == ["climatology", "persistence"]
    assert args.references == ["glorys", "glo12"]
    assert args.region == "global"
    assert args.year == 2024
    assert args.template_challenger == "glonet"
    assert args.start_limit is None


def test_build_pack_reproduces_the_old_quick_one_degree_pack():
    args = _parse(
        "build-pack",
        "/tmp/pack-quick-2024",
        "--kind",
        "quick",
        "--resolution",
        "one_degree",
        "--template-challenger",
        "glonet_1_degree",
        "--start-limit",
        "2",
        "--baselines",
    )

    assert args.kind == "quick"
    assert args.resolution == "one_degree"
    assert args.template_challenger == "glonet_1_degree"
    assert args.start_limit == 2
    assert args.baselines == []


def test_build_pack_rejects_an_unknown_kind():
    with pytest.raises(SystemExit):
        _parse("build-pack", "/tmp/pack", "--kind", "medium")


def test_build_pack_accepts_the_staging_flags():
    args = _parse("build-pack", "/tmp/pack", "--stage", "all", "--stage-dir", "/scratch/stage", "--cache-dir", "/c")

    assert args.stage == ["all"]
    assert args.stage_dir == "/scratch/stage"
    assert args.cache_dir == "/c"


def test_fetch_pack_defaults_to_the_cache():
    args = _parse("fetch-pack", "https://example.test/packs/pack-full-2024-global/")

    assert args.command == "fetch-pack"
    assert args.source == "https://example.test/packs/pack-full-2024-global/"
    assert args.dest is None


def test_fetch_pack_accepts_an_explicit_destination():
    args = _parse("fetch-pack", "https://example.test/packs/pack-a/", "--dest", "/tmp/packs/pack-a")

    assert args.dest == "/tmp/packs/pack-a"


def test_view_defaults_to_port_8799():
    args = _parse("view", "./oceanbench-evaluation/viewer/data")

    assert args.command == "view"
    assert args.artifacts_directory == "./oceanbench-evaluation/viewer/data"
    assert args.port == 8799


def test_evaluate_accepts_a_url_for_offline_references():
    args = _parse("evaluate", "my.zarr", "--offline-references", "https://example.test/packs/pack-a/")

    assert args.offline_references == "https://example.test/packs/pack-a/"
