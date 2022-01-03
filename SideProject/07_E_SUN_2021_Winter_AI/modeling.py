import lightgbm as lgbm
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", module="lightgbm")
import gc
import logging

logger = logging.getLogger(__name__)

class Model:
  def __init__(self, matrix):
    self.matrix = matrix
  
  def _pre_process(self):
    surplus_columns = [
        "txn_cnt",
        "slam",
        "masts",
        "educd",
        "trdtp",
        "naty",
        "poscd",
        "gender_code"
    ]
    self.matrix = self.matrix.drop(columns=surplus_columns)
    self.matrix['txn_amt'] = self.matrix['txn_amt'].clip(0, 20000)
  
  def _fit_booster(self,
      X_train,
      y_train,
      X_test=None,
      y_test=None,
      params=None,
      test_run=False,
      categoricals=[],
      dropcols=[],
      early_stopping=True,
    ):
    if params is None:
        params = {"learning_rate": 0.1,
                  "subsample_for_bin": 300000, "n_estimators": 50}

    early_stopping_rounds = None
    if early_stopping == True:
        early_stopping_rounds = 30

    if test_run:
        eval_set = [(X_train, y_train)]
    else:
        eval_set = [(X_train, y_train), (X_test, y_test)]

    booster = lgbm.LGBMRegressor(**params)

    categoricals = [c for c in categoricals if c in X_train.columns]

    booster.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        eval_metric=["rmse"],
        verbose=100,
        categorical_feature=categoricals,
        early_stopping_rounds=early_stopping_rounds,
    )
    return booster
    
  def _build_model(self):
    params = {
        "num_leaves": 966,
        "cat_smooth": 45.01680827234465,
        "min_child_samples": 27,
        "min_child_weight": 0.021144950289224463,
        "max_bin": 214,
        "learning_rate": 0.01,
        "subsample_for_bin": 300000,
        "min_data_in_bin": 7,
        "colsample_bytree": 0.8,
        "subsample": 0.6,
        "subsample_freq": 5,
        "n_estimators": 8000,
    }
    categoricals = [
        'month',
        'shop_tag_cluster',
        'trdtp_cluster',
        'age',
        'cuorg'
    ]  # These features will be set as categorical features by LightGBM and handled differently
    keep_from_month = 2  # The first couple of months are dropped because of distortions to their features (e.g. wrong item age)
    test_month = 24
    dropcols = [
        "chid",
        "shop_tag",
    ]  # The features are dropped to reduce overfitting
    valid = self.matrix.drop(columns=dropcols).loc[self.matrix.dt == test_month, :]
    train = self.matrix.drop(columns=dropcols).loc[self.matrix.dt < test_month, :]
    train = train[train.dt >= keep_from_month]
    X_train = train.drop(columns="txn_amt")
    y_train = train.txn_amt
    X_valid = valid.drop(columns="txn_amt")
    y_valid = valid.txn_amt
    lgbooster = self._fit_booster(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        params=params,
        test_run=False,
        categoricals=categoricals,
    )
    return lgbooster

def _model_predict(self, lgbooster):
  keep_from_month = 2
  test_month = 34
  test = matrix.loc[matrix.date_block_num == test_month, :]
