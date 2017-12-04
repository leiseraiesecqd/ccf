import time
import scipy as sp
from models import utils, parameters
from models.training_mode import TrainingMode


class Training:

    def __init__(self):
        pass

    def rmse(self,y_prob, y):
        return sp.sqrt(sp.mean((y_prob - y) ** 2))

    @staticmethod
    def get_cv_args(model_name=None):

        from models.cross_validation import CrossValidation

        if model_name == 'xgb':
            cv_args = {'n_splits': 5,
                       'n_cv': 5,
                       'cv_generator': CrossValidation. random_split}

        else:
            cv_args = {'n_splits': 5,
                       'n_cv': 5}
            print('------------------------------------------------------')
            print('[W] Training with Base cv_args:\n', cv_args)

        return cv_args

    @staticmethod
    def train_diff_round(model_name, TM, num_boost_round_list=None, n_epoch=1, full_grid_search=False,
                         train_seed_list=None, cv_seed_list=None, base_parameters=None, parameter_grid_list=None,
                         save_final_pred=True, train_args=None, cv_args=None):

        for num_boost_round in num_boost_round_list:
            append_info = train_args['append_info']
            train_args['append_info'] = append_info + '_' + str(num_boost_round)
            TM.auto_train_boost_round(model_name=model_name, num_boost_round=num_boost_round, n_epoch=n_epoch,
                                      full_grid_search=full_grid_search, train_seed_list=train_seed_list,
                                      cv_seed_list=cv_seed_list, base_parameters=base_parameters,
                                      parameter_grid_list=parameter_grid_list, save_final_pred=save_final_pred,
                                      train_args=train_args, cv_args=cv_args)
            train_args['append_info'] = append_info

    def train(self):
        """
            ## Auto Train with Logs of Boost Round ##

            Model Name:
            'lr':           Logistic Regression
            'rf':           Random Forest
            'et':           Extra Trees
            'gb':           GradientBoosting
            'xgb':          XGBoost
            'xgb_sk':       XGBoost using scikit-learn module
            'lgb':          LightGBM
            'lgb_sk':       LightGBM using scikit-learn module
            'cb':           CatBoost
        """
        TM = TrainingMode()

        """
            Training Arguments
        """
        train_args = {'use_global_valid': False,
                      'use_custom_obj': False,
                      'show_importance': False,
                      'save_final_pred': True,
                      'save_final_pred_train': False,
                      'save_cv_pred': False,
                      'save_cv_pred_train': False,
                      'save_csv_log': True,
                      'loss_fuc': self.rmse,
                      'append_info': 'Yuanan Bike'}

        """
            Cross Validation Arguments
        """
        cv_args = {'n_cv': 10
                   }

        # cv_args = self.get_cv_args('xgb')

        """
            Base Parameters
        """
        # base_parameters = self.get_base_params('xgb')
        base_parameters = None

        """
            Auto Train with Logs of Boost Round
        """
        pg_list = [
                   # [
                   #  ['n_cv', [5,7,9]]
                   #  ['valid_rate', valid_rate_list]
                   # ],
                     [
                      ['n_cv', [5,7,9]],
                      ['max_depth', [12,13,14]],# best 14
                      ['learning_rate', [0.15,0.2,0.25]],#best 0.15
                     #['min_child_weight', [9]],
                      ['subsample', [0.75,0.8]], #best0.8
                      ['colsample_bytree', [0.7]],#best 0.7
                      ['colsample_bylevel', [0.77]],#best0.77
                     ]
                  ]
        train_seed_list = None
        cv_seed_list = None
        TM.auto_train_boost_round('xgb', num_boost_round=600, n_epoch=3, full_grid_search=True,
                                  train_seed_list=train_seed_list, cv_seed_list=cv_seed_list,
                                  base_parameters=base_parameters, parameter_grid_list=pg_list, save_final_pred=True,
                                  train_args=train_args, cv_args=cv_args)

        """Train Different Rounds"""
        # num_boost_round_list = [83, 85, 87]
        # self.train_diff_round('xgb', TM, num_boost_round_list=num_boost_round_list, n_epoch=1, full_grid_search=True,
        #                       train_seed_list=train_seed_list, cv_seed_list=cv_seed_list,
        #                       base_parameters=base_parameters, parameter_grid_list=pg_list, save_final_pred=True,
        #                       train_args=train_args, cv_args=cv_args)


def get_base_params(model_name=None):
    """
        Get Base Parameters
    """
    if model_name == 'xgb':
        """
            XGB
        """
        base_parameters = {'learning_rate': 0.003,
                           'gamma': 0.001,
                           'max_depth': 10,
                           'min_child_weight': 8,
                           'subsample': 0.92,
                           'colsample_bytree': 0.85,
                           'colsample_bylevel': 0.7,
                           'lambda': 0,
                           'alpha': 0,
                           'early_stopping_rounds': 10000,
                           'n_jobs': -1,
                           'objective': 'reg:linear',
                           'eval_metric': 'rmse'}

    elif model_name == 'lgb':
        """
            LGB
        """
        base_parameters = {'application': 'regression',
                           'boosting': 'gbdt',
                           'learning_rate': 0.003,
                           'num_leaves': 88,
                           'max_depth': 9,
                           'min_data_in_leaf': 2500,
                           'min_sum_hessian_in_leaf': 1e-3,
                           'feature_fraction': 0.6,
                           'feature_fraction_seed': 19,
                           'bagging_fraction': 0.8,
                           'bagging_freq': 5,
                           'bagging_seed': 1,
                           'lambda_l1': 0,
                           'lambda_l2': 0,
                           'min_gain_to_split': 0,
                           'max_bin': 225,
                           'min_data_in_bin': 5,
                           'metric': 'l2_root',
                           'num_threads': -1,
                           'verbosity': 1,
                           'early_stopping_rounds': 10000}

    else:
        print('------------------------------------------------------')
        print('[W] Training without Base Parameters')
        base_parameters = None

    return base_parameters


if __name__ == "__main__":

    start_time = time.time()

    # Check if directories exit or not
    utils.check_dir(parameters.path_list)

    print('======================================================')
    print('Start Training...')

    T = Training()
    T.train()

    print('------------------------------------------------------')
    print('All Tasks Done!')
    print('Total Time: {}s'.format(time.time() - start_time))
    print('======================================================')
