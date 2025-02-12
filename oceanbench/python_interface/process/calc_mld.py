from pathlib import Path

from oceanbench.core.process.calc_mld_core import calc_mld_core


def calc_mld(glonet_dataset_path: Path, lead: int, output_path: Path):
    return calc_mld_core(glonet_dataset_path, lead, output_path)
