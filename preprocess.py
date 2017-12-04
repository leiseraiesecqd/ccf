import time
import numpy as np
import pandas as pd
from models import utils
from sklearn.preprocessing import OneHotEncoder

train_csv_path = './inputs/train_rt.csv'
test_csv_path = './inputs/test_rt.csv'
preprocessed_path = './data/preprocessed_data/'
gan_prob_path = './data/gan_outputs/'


class DataPreProcess:

    def __init__(self, train_path, test_path, preprocess_path, cat_list=None,
                 use_global_valid=False, global_valid_rate=None):

        self.train_path = train_path
        self.test_path = test_path
        self.preprocess_path = preprocess_path
        self.x_train = pd.DataFrame()
        self.x_g_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.x_g_test = pd.DataFrame()
        self.id_test = pd.DataFrame()

        # Validation Set
        self.x_valid = np.array([])
        self.x_g_valid = np.array([])
        self.y_valid = np.array([])

        self.cat_list = cat_list
        self.drop_feature_list = []
        self.use_global_valid_ = use_global_valid
        self.global_valid_rate = global_valid_rate

        if cat_list is not None:
            self.g_train = pd.DataFrame()
            self.g_test = pd.DataFrame()
            self.g_cat_dict = {}

    # Load CSV Files Using Pandas
    def load_csv(self):

        train_f = pd.read_csv(self.train_path, header=0 )
        test_f = pd.read_csv(self.test_path, header=0)

        return train_f, test_f

    # Load Data Using Pandas
    def load_data(self):

        try:
            print('======================================================')
            print('Loading data...')
            train_f, test_f = self.load_csv()
        except Exception as e:
            print('Unable to read data: ', e)
            raise

        # Drop Unnecessary Columns
        self.x_train = train_f.drop(['date','number'], axis=1)
        self.y_train = train_f['number']
        self.x_test = test_f.drop(['ID','date'], axis=1)
        self.id_test = test_f['ID']

        print('------------------------------------------------------')
        print('Train Features: {}\n'.format(self.x_train.shape[1]),
              'Test Features: {}'.format(self.x_test.shape[1]))

        if self.cat_list is not None:
            column_list = list(self.x_train.columns)
            for i in self.cat_list:
                self.g_cat_dict[i] = column_list.index(i)
                self.x_g_train = self.x_train
                self.x_g_test = self.x_test

    # Convert pandas DataFrames to numpy arrays
    def convert_pd_to_np(self):

        print('======================================================')
        print('Converting pandas DataFrames to numpy arrays...')

        self.x_train = np.array(self.x_train, dtype=np.float64)
        self.y_train = np.array(self.y_train, dtype=np.float64)
        self.x_test = np.array(self.x_test, dtype=np.float64)
        self.id_test = np.array(self.id_test, dtype=int)

    # Convert Column 'group' to Dummies
    def convert_group_to_dummies(self, add_train_dummies=False):

        print('======================================================')
        print('Converting Categorical features of Train Set to Dummies...')

        train_lenth = len(self.x_train)
        data_mix = np.vstack((self.x_train, self.x_test))
        enc = OneHotEncoder(categorical_features=list(self.g_cat_dict.values()))
        data_transform = enc.fit_transform(data_mix).toarray()
        self.x_train = data_transform[:train_lenth, :]
        self.x_test = data_transform[train_lenth:, :]

        print('------------------------------------------------------')
        print('Train Features After OneHotEncoding : {}\n'.format(self.x_train.shape[1]),
              'Test Features After OneHotEncoding : {}'.format(self.x_test.shape[1]))

    # Spilt Validation Set by valid_rate
    def split_validation_set(self, valid_rate=None):

        print('======================================================')
        print('Splitting Validation Set by Valid Rate: {}'.format(valid_rate))

        train_index = []
        valid_index = []

        # Validation Set
        self.x_valid = self.x_train[valid_index]
        self.x_g_valid = self.x_g_train[valid_index]
        self.y_valid = self.y_train[valid_index]

        # Train Set
        self.x_train = self.x_train[train_index]
        self.x_g_train = self.x_g_train[train_index]
        self.y_train = self.y_train[train_index]

    # Save Data
    def save_data(self):

        print('======================================================')
        print('Saving Preprocessed Data...')
        utils.save_data_to_pkl(self.x_train, self.preprocess_path + 'x_train.p')
        utils.save_data_to_pkl(self.x_g_train, self.preprocess_path + 'x_g_train.p')
        utils.save_data_to_pkl(self.y_train, self.preprocess_path + 'y_train.p')
        utils.save_data_to_pkl(self.x_test, self.preprocess_path + 'x_test.p')
        utils.save_data_to_pkl(self.x_g_test, self.preprocess_path + 'x_g_test.p')
        utils.save_data_to_pkl(self.id_test, self.preprocess_path + 'id_test.p')



    # Save Validation Set
    def save_global_valid_set(self):

        print('======================================================')
        print('Saving Validation Set...')
        utils.save_data_to_pkl(self.x_valid, self.preprocess_path + 'x_global_valid.p')
        utils.save_data_to_pkl(self.x_g_valid, self.preprocess_path + 'x_g_global_valid.p')
        utils.save_data_to_pkl(self.y_valid, self.preprocess_path + 'y_global_valid.p')

    # Save Data Split by Era Distribution

    # Preprocess
    def preprocess(self):

        print('======================================================')
        print('Start Preprocessing...')

        start_time = time.time()

        # Load original data
        self.load_data()

        # Convert pandas DataFrames to numpy arrays
        self.convert_pd_to_np()

        #Converting Categorical features of Train Set to Dummies
        self.convert_group_to_dummies()

        # Spilt Validation Set by valid_rate
        if self.use_global_valid_:
            self.split_validation_set(valid_rate=self.global_valid_rate)
            self.save_global_valid_set()

        # Save Data to pickle files
        self.save_data()

        end_time = time.time()

        print('======================================================')
        print('Done!')
        print('Using {:.3}s'.format(end_time - start_time))
        print('======================================================')


if __name__ == '__main__':

    utils.check_dir(['./data/', preprocessed_path])

    preprocess_args = {'cat_list': ['week', 'week_era', 'station'],
                       'use_global_valid': False}

    DPP = DataPreProcess(train_csv_path, test_csv_path, preprocessed_path, **preprocess_args)
    DPP.preprocess()
