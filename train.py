import argparse

from loader import Loader
from trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process models for training.')
    parser.add_argument("-d", "--data_folder", help="Insert data folder path")
    parser.add_argument("-c", "--best_config_folder", help="Insert best config folder path")
    parser.add_argument("-pt", "--ckpt_folder", help="Insert ckpt folder path")
    parser.add_argument("-", "--model", help="Insert model name")
    parser.add_argument('-t', "--tuning", help='tuning_type')
    args = parser.parse_args()

    loader = Loader(args.data_folder, 0.2, False)
    trainer = Trainer(loader, args.best_config_folder, args.ckpt_folder)
    print("Begin Training....")
    trainer._train_best(args.model, args.tuning)

