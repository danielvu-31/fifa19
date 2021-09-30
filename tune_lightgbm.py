from tuning import HyperParamTuning
from loader import Loader
import ray.tune as tune


if __name__ == '__main__':
    tune_config_path = "config/model_tuning.yaml"
    data_folder = "data"
    result = "tune_result"
    ray_folder = "ray_results"
    dataloader = Loader(folder_path=data_folder, split_pct=0.2, save=True)
    tuner = HyperParamTuning(tune_config_path, result, ray_folder, dataloader)
    tuner.tuning("lightgbm")

