# Import libraries
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn import metrics
from Loader import Loader

class Trainer():
    def __init__(self, loader):
        self.loader = loader

    def train_xgboost(self, model):
        train, val = self.loader.prepare_train(False)
        lgb_train = lgb.Dataset(train.drop(["Overall"], axis=1), train["Overall"])
        lgb_val = lgb.Dataset(val.drop(["Overall"], axis=1), val["Overall"], reference=lgb_train)

        params = {'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'num_leaves': 40,
                    'learning_rate': 0.1,
                    'feature_fraction': 0.9
                    }

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=200,
                        valid_sets=[lgb_train, lgb_val],
                        valid_names=['train','valid'],
                        categorical_feature=self.loader.cat_var
                        )
    
    def train_decision_regressor(self, model):
        train, val = self.loader.prepare_train(True)
        x_train, y_train = train.drop(["Overall"], axis=1), train["Overall"]
        x_val, y_val = val.drop(["Overall"], axis=1), val["Overall"]


# def auc2(m, train, test): 
#     return (metrics.roc_auc_score(y_train,m.predict(train)),
#                             metrics.roc_auc_score(y_test,m.predict(test)))

# lg = lgb.LGBMClassifier(silent=False)
# param_dist = {"max_depth": [25,50, 75],
#               "learning_rate" : [0.01,0.05,0.1],
#               "num_leaves": [300,900,1200],
#               "n_estimators": [200]
#              }
# grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="roc_auc", verbose=5)
# grid_search.fit(train,y_train)
# grid_search.best_estimator_

# d_train = lgb.Dataset(train, label=y_train)
# params = {"max_depth": 50, "learning_rate" : 0.1, "num_leaves": 900,  "n_estimators": 300}

# # Without Categorical Features
# model2 = lgb.train(params, d_train)
# auc2(model2, train, test)

# #With Catgeorical Features
# cate_features_name = ["MONTH","DAY","DAY_OF_WEEK","AIRLINE","DESTINATION_AIRPORT",
#                  "ORIGIN_AIRPORT"]
# model2 = lgb.train(params, d_train, categorical_feature = cate_features_name)
# auc2(model2, train, test)
