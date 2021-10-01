import argsparse

from tuning import HyperParamTuning
from loader import Loader
import ray.tune as tune

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning models...')
    parser.add_argument("--tune_config_path", help="Insert tune_config_path")
    parser.add_argument("--data_folder", help="Insert data_folder path")
    parser.add_argument("--tune_result", help="Insert tune_result save path")
    parser.add_argument("--model", help="Insert model name")
    parser.add_argument("--ray_results", help="Insert ray folder to save output")
    args = parser.parse_args()


    dataloader = Loader(folder_path=args.data_folder, split_pct=0.2, save=True)
    tuner = HyperParamTuning(args.tune_config_path, args.result, args.ray_folder, dataloader)
    tuner.tuning(args.model)

