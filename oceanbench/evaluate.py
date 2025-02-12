from pathlib import Path
from typing import Any

import numpy

from oceanbench.core.evaluate.rmse_core import glonet_pointwise_evaluation_core


def pointwise_evaluation(glonet_datasets_path: Path | str, glorys_datasets_path: Path | str) -> numpy.ndarray[Any]:
    if isinstance(glonet_datasets_path, str):
        glonet_datasets_path = Path(glonet_datasets_path)
    if isinstance(glorys_datasets_path, str):
        glorys_datasets_path = Path(glorys_datasets_path)
    return glonet_pointwise_evaluation_core(
        glonet_datasets_path=glonet_datasets_path,
        glorys_datasets_path=glorys_datasets_path,
    )
