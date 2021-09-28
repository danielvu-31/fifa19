import os
import json
import yaml
import lightgbm as lgb
import xgboost as xgb

import ray.tune as tune
from tune_sklearn import TuneSearchCV, TuneGridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor


def init_dict(nested_dictionary):
    for key, value in nested_dictionary.items():
        if isinstance(value, dict):
            init_dict(value)
        else:
            try:
                value = eval(value)
                nested_dictionary.update({key: value})
            except:
                continue
    return nested_dictionary


class HyperParamTuning():
    def __init__(self, tuning_config_path, result_path, loader):
        self.result_path = result_path
        with open(tuning_config_path) as file:
            tuning_config = yaml.load(file, Loader=yaml.FullLoader)
        self.tuning_config = init_dict(tuning_config)
        self.model_init = {
            "decision_tree": DecisionTreeRegressor(random_state=42),
            "random_forest": RandomForestRegressor(random_state=42),
            "extra_tree": ExtraTreeRegressor(random_state=42),
            "lightgbm": lgb.LGBMRegressor(random_state=42),
            "xgboost": xgb.XGBRegressor(random_state=42)
        }
        self.loader = loader
    
    def _init_config(self, model_name):
        model = self.model_init(model_name)
        fixed_param = self.tuning_config[model_name]["fixed_params"]
        for k, v in fixed_param:
            setattr(model, k, v)

        tuned_params = self.tuning_config[model_name]["params"]
        one_hot = self.tuning_config[model_name]["one_hot"]
        train, val = self.loader.prepare_train(one_hot)
        train = [train.drop(["Overall"], axis=1), train["Overall"]]
        val = [val.drop(["Overall"], axis=1), val["Overall"]]
        return model, (train, val), tuned_params

    def _extend_cat_arg(self, model_name):
        args = {}
        if not self.tuning_config[model_name]["one_hot"]:
            args["categorical_feature"] = self.loader.cat_var
        return args

    def tuning(self, model_name):
        model, (train, _), tuned_params = self._init_config(model_name)
        tuning_type = self.tuning_config[model_name]["tuning"]
        args = self._extend_cat_arg(model_name)

        if tuning_type == "gridsearch":
            searcher = TuneGridSearchCV(estimator=model,
                                        param_grid=tuned_params,
                                        scoring="neg_mean_absolute_error",
                                        verbose=2,
                                        mode="min",
                                        max_iters=10)
        else:
            searcher = TuneSearchCV(estimator=model,
                                      search_optimization=tuning_type,
                                      param_distributions=tuned_params,
                                      scoring="neg_mean_absolute_error",
                                      verbose=2,
                                      mode="min",
                                      max_iters=10)
    
        # train[0]: x_train
        # train[1]: y_train
        searcher.fit(train[0], train[1], **args)

        # Write best config to json
        self.save_output(f"{model_name}_{tuning_type}", 
                         self.tuning_config[model_name]["fixed_params"],
                         searcher.best_params_)
        return searcher.best_params_

    # Save best config to json
    def save_output(self, name, fixed_param, best_tuned_param):
        f = open(os.path.join(self.result_path, "best_params.json"), "r+")
        params_group = json.load(f)
        params_group[name] = {**fixed_param, **best_tuned_param}
        json.dump(params_group, f)
        f.close()

    