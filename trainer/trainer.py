# Import libraries
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn import metrics
import yaml
import json
import os
from joblib import dump

class Trainer():
    def __init__(self, loader, best_config_path, ckpt_folder):
        self.loader = loader
        with open(best_config_path) as file:
            self.param_config = json.load(file)
        # Hyperparam tuner class
        self.model_init = {
            "decision_tree": {
                "model": DecisionTreeRegressor(),
                "one_hot": True
                },
            "random_forest": {
                "model": RandomForestRegressor(),
                "one_hot": True
                },
            "extra_tree": {
                "model": ExtraTreeRegressor(),
                "one_hot": True
                },
            "lightgbm": {
                "model": lgb.LGBMRegressor(),
                "one_hot": False
                }
        }

        self.ckpt_folder = ckpt_folder
        if not os.path.exists(self.ckpt_folder):
            os.makedirs(self.ckpt_folder)

    def _init_config(self, model_name, tuning_type):
        assert f"{model_name}_{tuning_type}" in self.param_config.keys()

        best_param = self.param_config[f"{model_name}_{tuning_type}"]
        model = self.model_init[model_name]["model"]
        format_cat = self.model_init[model_name]["one_hot"]

        for k, v in best_param.items():
            setattr(model, k, v)

        train, val = self.loader.prepare_train(format_cat)
        train = [train.drop(["Overall"], axis=1), train["Overall"]]
        val = [val.drop(["Overall"], axis=1), val["Overall"]]

        return model, (train, val)
    
    def _extend_cat_arg(self, model_name):
        args = {}
        if not self.model_init[model_name]["one_hot"]:
            args["categorical_feature"] = self.loader.cat_var
        return args

    def _train_best(self, model_name, tuning_type):
        print(f"Fitting {model_name} Model....")
        model, (train, val) = self._init_config(model_name, tuning_type)
        args = self._extend_cat_arg(model_name)

        model.fit(train[0], train[1], **args)
        y_train = model.predict(train[0])
        train_result ={
            "l2": metrics.mean_squared_error(train[1], y_train),
            "abs_error": metrics.mean_absolute_error(train[1], y_train),
            "r2": metrics.r2_score(train[1], y_train)
        }
        print("Train Results:")
        print("  - L2: {:.3f}".format(float(train_result["l2"])))
        print("  - MAE: {:.3f}".format(float(train_result["abs_error"])))                                                       
        print("  - R2: {:.3f}".format(float(train_result["r2"])))

        y_pred = model.predict(val[0])
        result ={
            "l2": metrics.mean_squared_error(val[1], y_pred),
            "abs_error": metrics.mean_absolute_error(val[1], y_pred),
            "r2": metrics.r2_score(val[1], y_pred)
        }
        print("Validation Results:")
        print("  - L2: {:.3f}".format(float(result["l2"])))
        print("  - MAE: {:.3f}".format(float(result["abs_error"])))                                                       
        print("  - R2: {:.3f}".format(float(result["r2"])))

        # Save model ckpt
        path = os.path.join(self.ckpt_folder, f"{model_name}_{tuning_type}_ckpt.joblibs")
        dump(model, path) 
        # Save json file
        self._save_output(model_name, tuning_type, result, path)
    
    def _save_output(self, model_name, tuning, result, path):
        output = {
            "name": model_name,
            "tuning_type": tuning,
            "results": result,
            "ckpt_path": path
        }

        # Dump json output
        json_path = os.path.join(self.ckpt_folder, f"{model_name}_{tuning}_results.json")
        with open(json_path, "w") as outfile:
            json.dump(output, outfile)

