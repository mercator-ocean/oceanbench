from pathlib import Path

from oceanbench.core.evaluate.rmse_core import glonet_pointwise_evaluation_core


def pointwise_evaluation(glonet_datasets_path: Path, glorys_datasets_path: Path, output_rmse: Path):
    return glonet_pointwise_evaluation_core(glonet_datasets_path, glorys_datasets_path, output_rmse)
