import preprocess
from models.regressors import *
from models.sk_grid_search import SKLearnGridSearch


pred_path = './results/'
single_model_pred_path = pred_path + 'single_model/'
prejudge_pred_path = pred_path + 'prejudge/'
stack_pred_path = pred_path + 'stacking/'
auto_train_pred_path = pred_path + 'auto_train/'
log_path = './logs/'
csv_log_path = './logs/csv_logs/'
grid_search_out_path = './grid_search_outputs/'
boost_round_out_path = './boost_round_outputs/'
loss_log_path = log_path + 'loss_logs/'
prejudge_loss_log_path = loss_log_path + 'prejudge/'
dnn_log_path = log_path + 'dnn_logs/'
grid_search_log_path = log_path + 'grid_search_logs/'
data_path = './data/'
preprocessed_data_path = preprocess.preprocessed_path
prejudged_data_path = data_path + 'prejudged_data/'
stack_output_path = data_path + 'stacking_outputs/'
model_checkpoint_path = './checkpoints/'
dnn_checkpoint_path = model_checkpoint_path + 'dnn_checkpoints/'


path_list = [pred_path,
             single_model_pred_path,
             prejudge_pred_path,
             stack_pred_path,
             auto_train_pred_path,
             log_path,
             csv_log_path,
             grid_search_out_path,
             grid_search_log_path,
             boost_round_out_path,
             prejudge_loss_log_path,
             loss_log_path,
             dnn_log_path,
             data_path,
             prejudged_data_path,
             stack_output_path,
             model_checkpoint_path,
             dnn_checkpoint_path]


class SingleModel:
    """
        Train single model
    """
    def __init__(self, reduced_feature_list=None, train_args=None, cv_args=None, mode=None):

        self.x_train, self.y_train,  self.x_test, self.id_test = \
            utils.load_preprocessed_data(preprocessed_data_path)
        self.x_g_train, self.x_g_test = \
            utils.load_preprocessed_data_g(preprocessed_data_path)

        if train_args['use_global_valid']:
            self.x_gl_valid, self.x_g_gl_valid, self.y_gl_valid = \
                utils.load_global_valid_data(preprocessed_data_path)

        # Choose Useful features
        if reduced_feature_list is not None:

            reduced_feature_list_g = reduced_feature_list + [-1]
            reduced_feature_list.extend(list(range(-1, -29, -1)))
            self.x_train = self.x_train[:, reduced_feature_list]
            self.x_g_train = self.x_g_train[:, reduced_feature_list_g]
            self.x_test = self.x_test[:, reduced_feature_list]
            self.x_g_test = self.x_g_test[:, reduced_feature_list_g]
            if train_args['use_global_valid']:
                self.x_gl_valid = self.x_gl_valid[reduced_feature_list]
                self.x_g_gl_valid = self.x_g_gl_valid[reduced_feature_list_g]

        self.loss_log_path = loss_log_path
        self.boost_round_log_path = boost_round_out_path
        self.train_args = train_args
        self.cv_args = cv_args
        self.mode = mode

        # Different Mode
        if mode == 'auto_grid_search':
            self.csv_log_path = grid_search_out_path
            self.pred_path = single_model_pred_path
        elif mode == 'auto_train':
            self.csv_log_path = auto_train_pred_path
            self.pred_path = auto_train_pred_path
        else:
            self.csv_log_path = csv_log_path + 'single_'
            self.pred_path = single_model_pred_path

    def train_model(self, model=None, grid_search_tuple_list=None):

        if grid_search_tuple_list is not None:
            param_name_list = []
            param_value_list = []
            for grid_search_tuple in grid_search_tuple_list:
                param_name = grid_search_tuple[0]
                param_value = grid_search_tuple[1]
                if param_name in ['n_valid', 'n_cv', 'n_era', 'valid_rate', 'window_size', 'cv_weights']:
                    self.cv_args[param_name] = param_value
                else:
                    self.train_args['parameters'][param_name] = param_value
                param_name_list.append(param_name)
                param_value_list.append(param_value)

        else:
            param_name_list = None
            param_value_list = None

        # Parameters for Train
        model.train(pred_path=self.pred_path, loss_log_path=self.loss_log_path, csv_log_path=self.csv_log_path,
                    boost_round_log_path=self.boost_round_log_path, mode=self.mode, param_name_list=param_name_list,
                    param_value_list=param_value_list, cv_args=self.cv_args, **self.train_args)

    def lr_train(self, train_seed, cv_seed, parameters=None, grid_search_tuple_list=None):
        """
            Logistic Regression
        """
        if parameters is None:
            parameters = {'C': 1.0,
                          'class_weight': None,
                          'dual': False,
                          'fit_intercept': True,
                          'intercept_scaling': 1,
                          'max_iter': 100,
                          # 'multi_class': 'multinomial',
                          'multi_class': 'ovr',
                          'n_jobs': -1,
                          'penalty': 'l2',
                          'solver': 'sag',
                          'tol': 0.0001,
                          'random_state': train_seed,
                          'verbose': 1,
                          'warm_start': False}
        else:
            parameters['random_state'] = train_seed

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed

        if self.train_args['use_global_valid']:
            model = LRegression(self.x_train, self.y_train, self.x_test, self.id_test, self.x_gl_valid, self.y_gl_valid)
        else:
            model = LRegression(self.x_train, self.y_train, self.x_test, self.id_test)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)

    def rf_train(self, train_seed, cv_seed, parameters=None, grid_search_tuple_list=None):
        """
            Random Forest
        """
        if parameters is None:
            parameters = {'bootstrap': True,
                          'class_weight': None,
                          'criterion': 'gini',
                          'max_depth': 2,
                          'max_features': 7,
                          'max_leaf_nodes': None,
                          'min_impurity_decrease': 0.0,
                          'min_samples_leaf': 286,
                          'min_samples_split': 3974,
                          'min_weight_fraction_leaf': 0.0,
                          'n_estimators': 32,
                          'n_jobs': -1,
                          'oob_score': True,
                          'random_state': train_seed,
                          'verbose': 2,
                          'warm_start': False}
        else:
            parameters['random_state'] = train_seed

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed

        if self.train_args['use_global_valid']:
            model = RandomForest(self.x_train, self.y_train, self.x_test, self.id_test, self.x_gl_valid, self.y_gl_valid)
        else:
            model = RandomForest(self.x_train, self.y_train, self.x_test, self.id_test)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)

    def et_train(self, train_seed, cv_seed, parameters=None, grid_search_tuple_list=None):
        """
            Extra Trees
        """
        if parameters is None:
            parameters = {'bootstrap': True,
                          'class_weight': None,
                          'criterion': 'gini',
                          'max_depth': 2,
                          'max_features': 7,
                          'max_leaf_nodes': None,
                          'min_impurity_decrease': 0.0,
                          'min_samples_leaf': 357,
                          'min_samples_split': 4909,
                          'min_weight_fraction_leaf': 0.0,
                          'n_estimators': 20,
                          'n_jobs': -1,
                          'oob_score': True,
                          'random_state': train_seed,
                          'verbose': 2,
                          'warm_start': False}
        else:
            parameters['random_state'] = train_seed

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed

        if self.train_args['use_global_valid']:
            model = ExtraTrees(self.x_train, self.y_train, self.x_test, self.id_test, self.x_gl_valid, self.y_gl_valid)
        else:
            model = ExtraTrees(self.x_train, self.y_train, self.x_test, self.id_test)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)

    def gb_train(self, train_seed, cv_seed, parameters=None, grid_search_tuple_list=None):
        """
            GradientBoosting
        """
        if parameters is None:
            parameters = {'criterion': 'friedman_mse',
                          'init': None,
                          'learning_rate': 0.05,
                          'loss': 'deviance',
                          'max_depth': 25,
                          'max_features': 'auto',
                          'max_leaf_nodes': None,
                          'min_impurity_decrease': 0.0,
                          'min_impurity_split': None,
                          'min_samples_leaf': 50,
                          'min_samples_split': 1000,
                          'min_weight_fraction_leaf': 0.0,
                          'n_estimators': 200,
                          'presort': 'auto',
                          'random_state': train_seed,
                          'subsample': 0.8,
                          'verbose': 2,
                          'warm_start': False}
        else:
            parameters['random_state'] = train_seed

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed

        if self.train_args['use_global_valid']:
            model = GradientBoosting(self.x_train, self.y_train, self.x_test, self.id_test, self.x_gl_valid, self.y_gl_valid)
        else:
            model = GradientBoosting(self.x_train, self.y_train, self.x_test, self.id_test)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)

    def xgb_train(self, train_seed, cv_seed, parameters=None, grid_search_tuple_list=None, num_boost_round=None):
        """
            XGBoost
        """
        if parameters is None:
            parameters = {'learning_rate': 0.003,
                          'gamma': 0.001,                   # 如果loss function小于设定值，停止产生子节点
                          'max_depth': 10,                  # default=6
                          'min_child_weight': 12,           # default=1，建立每个模型所需最小样本权重和
                          'subsample': 0.92,                # 建立树模型时抽取子样本占整个样本的比例
                          'colsample_bytree': 0.85,         # 建立树时对特征随机采样的比例
                          'colsample_bylevel': 0.7,
                          'lambda': 0,
                          'alpha': 0,
                          'early_stopping_rounds': 10000,
                          'n_jobs': -1,
                          'objective': 'reg:linear',
                          'eval_metric': 'rmse',
                          'seed': train_seed}
        else:
            parameters['seed'] = train_seed

        file_name_params = ['num_boost_round', 'learning_rate', 'max_depth', 'subsample',
                            'colsample_bytree', 'colsample_bylevel', 'gamma']

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed
        self.train_args['file_name_params'] = file_name_params

        if num_boost_round is None:
            num_boost_round = 150

        if self.train_args['use_global_valid']:
            model = XGBoost(self.x_train, self.y_train, self.x_test, self.id_test, self.x_gl_valid, 
                            self.y_gl_valid, num_boost_round=num_boost_round)
        else:
            model = XGBoost(self.x_train, self.y_train, self.x_test, self.id_test, num_boost_round=num_boost_round)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)

    def xgb_train_sklearn(self, train_seed, cv_seed, parameters=None, grid_search_tuple_list=None):
        """
            XGBoost using scikit-learn module
        """
        if parameters is None:
            parameters = {'max_depth': 3,
                          'learning_rate': 0.1,
                          'n_estimators': 100,
                          'silent': True,
                          'objective': "reg:linear",
                          #  'booster': 'gbtree',
                          #  'n_jobs':  1,
                          'nthread': -1,
                          'gamma': 0,
                          'min_child_weight': 1,
                          'max_delta_step': 0,
                          'subsample': 1,
                          'colsample_bytree': 1,
                          'colsample_bylevel': 1,
                          'reg_alpha': 0,
                          'reg_lambda': 1,
                          'scale_pos_weight': 1,
                          'base_score': 0.5,
                          #  'random_state': train_seed,
                          'seed': train_seed,
                          'missing': None}
        else:
            parameters['seed'] = train_seed

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed

        if self.train_args['use_global_valid']:
            model = SKLearnXGBoost(self.x_train, self.y_train, self.x_test, self.id_test, self.x_gl_valid, self.y_gl_valid)
        else:
            model = SKLearnXGBoost(self.x_train, self.y_train, self.x_test, self.id_test)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)

    def lgb_train(self, train_seed, cv_seed, parameters=None, grid_search_tuple_list=None, num_boost_round=None):
        """
            LightGBM
        """
        if parameters is None:
            parameters = {'application': 'regression',
                          'boosting': 'gbdt',                   # gdbt,rf,dart,goss
                          'learning_rate': 0.003,               # default=0.1
                          'num_leaves': 88,                     # default=31       <2^(max_depth)
                          'max_depth': 9,                       # default=-1
                          'min_data_in_leaf': 2500,             # default=20       reduce over-fit
                          'min_sum_hessian_in_leaf': 1e-3,      # default=1e-3     reduce over-fit
                          'feature_fraction': 0.5,              # default=1
                          'feature_fraction_seed': 19,          # default=2
                          'bagging_fraction': 0.8,              # default=1
                          'bagging_freq': 2,                    # default=0        perform bagging every k iteration
                          'bagging_seed': 1,                    # default=3
                          'lambda_l1': 0,                       # default=0
                          'lambda_l2': 0,                       # default=0
                          'min_gain_to_split': 0,               # default=0
                          'max_bin': 225,                       # default=255
                          'min_data_in_bin': 5,                 # default=5
                          'metric': 'l2_root',
                          'num_threads': -1,
                          'verbosity': 1,
                          'early_stopping_rounds': 10000}
        else:
            parameters['seed'] = train_seed

        file_name_params = ['num_boost_round', 'learning_rate', 'num_leaves', 'max_depth',
                            'feature_fraction', 'bagging_fraction', 'bagging_freq', 'min_data_in_bin']

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed
        self.train_args['file_name_params'] = file_name_params

        if num_boost_round is None:
            num_boost_round = 100

        if self.train_args['use_global_valid']:
            model = LightGBM(self.x_g_train, self.y_train, self.x_g_test, self.id_test,
                             self.x_g_gl_valid, self.y_gl_valid, num_boost_round=num_boost_round)
        else:
            model = LightGBM(self.x_g_train, self.y_train, self.x_g_test, self.id_test, num_boost_round=num_boost_round)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)

    def lgb_train_sklearn(self, train_seed, cv_seed, parameters=None, grid_search_tuple_list=None):
        """
            LightGBM using scikit-learn module
        """
        if parameters is None:
            parameters = {'learning_rate': 0.003,
                          'boosting_type': 'gbdt',        # traditional Gradient Boosting Decision Tree.
                          'num_leaves': 80,               # <2^(max_depth)
                          'max_depth': 7,                 # default=-1
                          'n_estimators': 50,
                          'max_bin': 1005,
                          'subsample_for_bin': 1981,
                          'objective': 'regression',
                          'min_split_gain': 0.,
                          'min_child_weight': 1,
                          'min_child_samples': 0,
                          'subsample': 0.8,
                          'subsample_freq': 5,
                          'colsample_bytree': 0.8,
                          'reg_alpha': 0.5,
                          'reg_lambda': 0.5,
                          'silent': False,
                          'seed': train_seed}
        else:
            parameters['seed'] = train_seed

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed

        if self.train_args['use_global_valid']:
            model = SKLearnLightGBM(self.x_g_train, self.y_train, self.x_g_test, self.id_test,
                                    self.x_g_gl_valid, self.y_gl_valid)
        else:
            model = SKLearnLightGBM(self.x_g_train, self.y_train, self.x_g_test, self.id_test)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)

    def cb_train(self, train_seed, cv_seed, parameters=None, grid_search_tuple_list=None, num_boost_round=None):
        """
            CatBoost
        """
        if parameters is None:
            parameters = {'iterations': 70,
                          'learning_rate': 0.004,
                          'depth': 9,                            # Depth of the tree.
                          'l2_leaf_reg': 0.01,                   # L2 regularization coefficient.
                          'rsm': 0.8,                            # The percentage of features to use at each iteration.
                          'bagging_temperature': 0.7,            # Controls intensity of Bayesian bagging. The higher the temperature the more aggressive bagging is.
                          'loss_function': 'RMSE',
                          'border': 0.5,
                          'border_count': 128,
                          'feature_border_type': 'MinEntropy',
                          'fold_permutation_block_size': 1,
                          'od_pval': None,                       # Use overfitting detector to stop training when reaching a specified threshold.
                          'od_wait': None,                       # Number of iterations which overfitting detector will wait after new best error.
                          'od_type': 'IncToDec',                 # Type of overfitting detector which will be used in program.
                          'gradient_iterations': None,           # The number of gradient steps when calculating the values in leaves.
                          'leaf_estimation_method': 'Gradient',  # The method used to calculate the values in leaves.
                          'thread_count': None,                  # Number of parallel threads used to run CatBoost.
                          'random_seed': train_seed,
                          'use_best_model': False,               # To limit the number of trees in predict() using information about the optimal value of the error function.
                          'verbose': True,
                          'ctr_description': None,               # Binarization settings for categorical features.
                          'ctr_border_count': 16,                # The number of partitions for Categ features.
                          'ctr_leaf_count_limit': None,          # The maximum number of leafs with categorical features.
                          'priors': None,                        # Use priors when training.
                          'has_time': False,                     # To use the order in which objects are represented in the input data.
                          'name': 'experiment',
                          'ignored_features': None,
                          'train_dir': None,
                          'custom_loss': None,
                          'eval_metric': 'RMSE',
                          'class_weights': None}
        else:
            parameters['random_seed'] = train_seed

        if num_boost_round is not None:
            parameters['iterations'] = num_boost_round

        file_name_params = ['iterations', 'learning_rate', 'depth', 'rsm', 'bagging_temperature']

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed
        self.train_args['file_name_params'] = file_name_params

        if self.train_args['use_global_valid']:
            model = CatBoost(self.x_g_train, self.y_train, self.x_g_test, self.id_test, self.x_g_gl_valid, self.y_gl_valid)
        else:
            model = CatBoost(self.x_g_train, self.y_train, self.x_g_test, self.id_test)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)


class SKGridSearch:
    """
        Grid Search
    """
    def __init__(self):
        pass

    @staticmethod
    def lr_grid_search(train_seed, cv_generator, cv_args):
        """
            Logistic Regression
        """
        _log_path = grid_search_log_path + 'lr_'

        x_train, y_train, x_test, id_test = utils.load_preprocessed_data(preprocessed_data_path)

        parameters = {'C': 1.0,
                      'class_weight': None,
                      'dual': False,
                      'fit_intercept': 'True',
                      'intercept_scaling': 1,
                      'max_iter': 100,
                      'multi_class': 'multinomial',
                      'n_jobs': -1,
                      'penalty': 'l2',
                      'solver': 'sag',
                      'tol': 0.0001,
                      'random_state': train_seed,
                      'verbose': 2,
                      'warm_start': False}

        LR = LRegression(x_train, y_train, x_test, id_test)

        reg = LR.get_reg(parameters)

        # parameters_grid = None

        parameters_grid = {
                           'C': (0.2, 0.5, 1),
                           'max_iter': (50, 100, 200),
                           'tol': (0.001, 0.005, 0.01)
                           }

        SKLearnGridSearch.grid_search(_log_path, x_train, y_train, reg, params=parameters,
                                      params_grid=parameters_grid, cv_generator=cv_generator, cv_args=cv_args)

        utils.print_grid_info('Logistic Regression', parameters, parameters_grid)

    @staticmethod
    def rf_grid_search(train_seed, cv_generator, cv_args):
        """
            Random Forest
        """
        _log_path = grid_search_log_path + 'rf_'

        x_train, y_train, x_test, id_test = utils.load_preprocessed_data(preprocessed_data_path)

        parameters = {'n_estimators': 32,
                      'bootstrap': True,
                      'class_weight': None,
                      'criterion': 'gini',
                      'max_depth': 6,
                      'max_features': 7,
                      'max_leaf_nodes': None,
                      'min_impurity_decrease': 0.0,
                      'min_samples_leaf': 300,
                      'min_samples_split': 4000,
                      'min_weight_fraction_leaf': 0.0,
                      'n_jobs': -1,
                      'oob_score': True,
                      'random_state': train_seed,
                      'verbose': 2,
                      'warm_start': False}

        RF = RandomForest(x_train, y_train, x_test, id_test)

        reg = RF.get_reg(parameters)

        # parameters_grid = None

        parameters_grid = {
                           # 'n_estimators': (30, 31, 32),
                           'max_depth': (2, 3),
                           # 'max_features': (6, 7),
                           'min_samples_leaf': (286, 287),
                           'min_samples_split': (3972, 3974, 3976, 3978)
                           }

        SKLearnGridSearch.grid_search(_log_path, x_train, y_train, reg, params=parameters,
                                      params_grid=parameters_grid, cv_generator=cv_generator, cv_args=cv_args)

        utils.print_grid_info('Random Forest', parameters, parameters_grid)

    @staticmethod
    def et_grid_search(train_seed, cv_generator, cv_args):
        """
            Extra Trees
        """
        _log_path = grid_search_log_path + 'et_'

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_data(
            preprocessed_data_path)

        parameters = {'bootstrap': True,
                      'n_estimators': 50,
                      'class_weight': None,
                      'criterion': 'gini',
                      'max_depth': 25,
                      'max_features': 'auto',
                      'max_leaf_nodes': None,
                      'min_impurity_decrease': 0.0,
                      'min_samples_leaf': 50,
                      'min_samples_split': 1000,
                      'min_weight_fraction_leaf': 0.0,
                      'n_jobs': -1,
                      'oob_score': True,
                      'random_state': train_seed,
                      'verbose': 2,
                      'warm_start': False}

        ET = ExtraTrees(x_train, y_train, x_test, id_test)

        reg = ET.get_reg(parameters)

        # parameters_grid = None

        parameters_grid = {
                           'n_estimators': (30, 40, 50),
                           'max_depth': (5, 6),
                           'max_features': (6, 7),
                           'min_samples_leaf': (200, 250, 300),
                           'min_samples_split': (3000, 3500, 4000)
                           }

        SKLearnGridSearch.grid_search(_log_path, x_train, y_train, reg, params=parameters,
                                      params_grid=parameters_grid, cv_generator=cv_generator, cv_args=cv_args)

        utils.print_grid_info('Extra Trees', parameters, parameters_grid)

    @staticmethod
    def gb_grid_search(train_seed, cv_generator, cv_args):
        """
            GradientBoosting
        """
        _log_path = grid_search_log_path + 'gb_'

        x_train, y_train, x_test, id_test = utils.load_preprocessed_data(preprocessed_data_path)

        parameters = {'criterion': 'friedman_mse',
                      'init': None,
                      'learning_rate': 0.05,
                      'loss': 'deviance',
                      'max_depth': 25,
                      'max_features': 'auto',
                      'max_leaf_nodes': None,
                      'min_impurity_decrease': 0.0,
                      'min_impurity_split': None,
                      'min_samples_leaf': 50,
                      'min_samples_split': 1000,
                      'min_weight_fraction_leaf': 0.0,
                      'n_estimators': 200,
                      'presort': 'auto',
                      'random_state': train_seed,
                      'subsample': 0.8,
                      'verbose': 2,
                      'warm_start': False}

        GB = GradientBoosting(x_train, y_train, x_test, id_test)

        reg = GB.get_reg(parameters)

        # parameters_grid = None

        parameters_grid = {
                           'n_estimators': (20, 50, 100),
                           'learning_rate': (0.05, 0.2, 0.5),
                           'max_depth': (5, 10, 15),
                           'max_features': (6, 8, 10),
                           'min_samples_leaf': (300, 400, 500),
                           'min_samples_split': (3000, 4000, 5000),
                           'subsample': (0.6, 0.8, 1)
                           }

        SKLearnGridSearch.grid_search(_log_path, x_train, y_train, reg, params=parameters,
                                      params_grid=parameters_grid, cv_generator=cv_generator, cv_args=cv_args)

        utils.print_grid_info('GradientBoosting', parameters, parameters_grid)

    @staticmethod
    def xgb_grid_search(train_seed, cv_generator, cv_args):
        """
            XGBoost
        """
        _log_path = grid_search_log_path + 'xgb_'

        x_train, y_train, x_test, id_test = utils.load_preprocessed_data(preprocessed_data_path)

        parameters = {'objective': 'reg:linear',
                      'learning_rate': 0.002,
                      'n_estimators': 100,
                      'max_depth': 9,
                      'min_child_weight': 5,
                      'max_delta_step': 0,
                      'silent': False,
                      'subsample': 0.8,
                      'colsample_bytree': 0.8,
                      'colsample_bylevel': 1,
                      'base_score': 0.5,
                      'gamma': 0,
                      'reg_alpha': 0,
                      'reg_lambda': 0,
                      'nthread': -1,
                      'seed': train_seed
                      # 'missing': None,
                      # 'nthread': -1,
                      # 'scale_pos_weight': 1,
                      }

        XGB = SKLearnXGBoost(x_train, y_train, x_test, id_test)

        reg = XGB.get_reg(parameters)

        # parameters_grid = None

        parameters_grid = {'learning_rate': (0.002, 0.005, 0.01),
                           'n_estimators': (20, 50, 100, 150),
                           'max_depth': (5, 7, 9),
                           # 'subsample': 0.8,
                           # 'colsample_bytree': 0.8,
                           # 'colsample_bylevel': 1,
                           # 'gamma': 0,
                           # 'min_child_weight': 1,
                           # 'max_delta_step': 0,
                           # 'base_score': 0.5,
                           # 'reg_alpha': 0,
                           # 'reg_lambda': 0,
                           }

        SKLearnGridSearch.grid_search(_log_path, x_train, y_train, reg, params=parameters,
                                      params_grid=parameters_grid, cv_generator=cv_generator, cv_args=cv_args)

        utils.print_grid_info('XGBoost', parameters, parameters_grid)

    @staticmethod
    def lgb_grid_search(train_seed, cv_generator, cv_args):
        """
            LightGBM
        """
        _log_path = grid_search_log_path + 'lgb_'

        x_train, y_train, x_test, id_test = utils.load_preprocessed_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_data_g(preprocessed_data_path)

        parameters = {'learning_rate': 0.006,
                      'boosting_type': 'gbdt',        # traditional Gradient Boosting Decision Tree.
                      # 'boosting_type': 'dart',        # Dropouts meet Multiple Additive Regression Trees.
                      # 'boosting_type': 'goss',        # Gradient-based One-Side Sampling.
                      # 'boosting_type': 'rf',          # Random Forest.
                      'num_leaves': 3,                # <2^(max_depth)
                      'max_depth': 8,                 # default=-1
                      'n_estimators': 79,
                      'max_bin': 1005,
                      'subsample_for_bin': 1981,
                      'objective': 'regression',
                      'min_split_gain': 0.,
                      'min_child_weight': 1,
                      'min_child_samples': 0,
                      'subsample': 0.723,
                      'subsample_freq': 3,
                      'colsample_bytree': 0.11,
                      'reg_alpha': 0.,
                      'reg_lambda': 0.,
                      'silent': False,
                      'seed': train_seed}

        LGB = SKLearnLightGBM(x_g_train, y_train, x_g_test, id_test)

        reg = LGB.get_reg(parameters)

        # parameters_grid = None

        parameters_grid = {
                           'learning_rate': (0.002, 0.005, 0.01),
                           'n_estimators': (30, 60, 90),
                           'num_leaves': (32, 64, 128),             # <2^(max_depth)
                           'colsample_bytree': (0.6, 0.8, 0.1),
                           'max_depth': (6, 8, 10),                 # default=-1
                           # 'min_data_in_leaf': 20,                  # default=20
                           # 'bagging_fraction': (0.5, 0.7, 0.9),
                           # 'feature_fraction': (0.5, 0.7, 0.9),
                           # 'subsample_for_bin': (50000, 100000, 150000),
                           # 'subsample_freq': (4, 6, 8),
                           # 'subsample': (0.6, 0.8, 1.0),
                           # 'max_bin': (255, 355, 455)
                           }

        SKLearnGridSearch.grid_search(_log_path, x_train, y_train, reg, params=parameters,
                                      params_grid=parameters_grid, cv_generator=cv_generator, cv_args=cv_args)

        utils.print_grid_info('LightGBM', parameters, parameters_grid)

    # Stacking Layer LightGBM
    @staticmethod
    def stack_lgb_grid_search(train_seed, cv_generator, cv_args):

        _log_path = grid_search_log_path + 'stk_lgb_'

        x_train, y_train, x_test, id_test = utils.load_preprocessed_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_data_g(preprocessed_data_path)
        blender_x_tree, blender_test_tree, blender_x_g_tree, blender_test_g_tree \
            = utils.load_stacked_data(stack_output_path + 'l2_')

        g_train = x_g_train[:, -1]
        x_train_reuse = x_train[:, :88]

        print('------------------------------------------------------')
        print('Stacking Reused Features of Train Set...')
        blender_x_tree = np.concatenate((blender_x_tree, x_train_reuse), axis=1)
        blender_x_g_tree = np.column_stack((blender_x_tree, g_train))

        parameters = {'learning_rate': 0.006,
                      'boosting_type': 'gbdt',        # traditional Gradient Boosting Decision Tree.
                      # 'boosting_type': 'dart',        # Dropouts meet Multiple Additive Regression Trees.
                      # 'boosting_type': 'goss',        # Gradient-based One-Side Sampling.
                      # 'boosting_type': 'rf',          # Random Forest.
                      'num_leaves': 3,  # <2^(max_depth)
                      'max_depth': 8,  # default=-1
                      'n_estimators': 79,
                      'max_bin': 1005,
                      'subsample_for_bin': 1981,
                      'objective': 'regression',
                      'min_split_gain': 0.,
                      'min_child_weight': 1,
                      'min_child_samples': 0,
                      'subsample': 0.723,
                      'subsample_freq': 3,
                      'colsample_bytree': 0.11,
                      'reg_alpha': 0.,
                      'reg_lambda': 0.,
                      'silent': False,
                      'random_state': train_seed}

        LGB = SKLearnLightGBM(blender_x_g_tree, y_train, blender_test_g_tree, id_test)

        reg = LGB.get_reg(parameters)

        # parameters_grid = None

        parameters_grid = {
            'learning_rate': (0.002, 0.005, 0.01),
            'n_estimators': (30, 60, 90),
            'num_leaves': (32, 64, 128),             # <2^(max_depth)
            'colsample_bytree': (0.6, 0.8, 0.1),
            'max_depth': (6, 8, 10),                 # default=-1
            # 'min_data_in_leaf': 20,                  # default=20
            # 'bagging_fraction': (0.5, 0.7, 0.9),
            # 'feature_fraction': (0.5, 0.7, 0.9),
            # 'subsample_for_bin': (50000, 100000, 150000),
            # 'subsample_freq': (4, 6, 8),
            # 'subsample': (0.6, 0.8, 1.0),
            # 'max_bin': (255, 355, 455)
        }

        SKLearnGridSearch.grid_search(_log_path, x_train, y_train, reg, params=parameters,
                                      params_grid=parameters_grid, cv_generator=cv_generator, cv_args=cv_args)

        utils.print_grid_info('LightGBM', parameters, parameters_grid)
