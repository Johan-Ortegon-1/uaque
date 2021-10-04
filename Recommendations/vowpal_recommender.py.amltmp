import logging
import numpy as np
import pandas as pd
import scrapbook as sb
from sklearn.preprocessing import minmax_scale
from numpy import interp
import time

from recommenders.utils.python_utils import binarize
from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
    mae,
    logloss,
    rsquared,
    exp_var
)
import sys
import gzip, pickle, pickletools

import sys

import os
from subprocess import run
from tempfile import TemporaryDirectory
from time import process_time

import pandas as pd
import papermill as pm
import scrapbook as sb
from numpy import interp

from recommenders.utils.notebook_utils import is_jupyter
from recommenders.datasets.movielens import load_pandas_df
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import (rmse, mae, exp_var, rsquared, get_top_k_items,
                                                     map_at_k, ndcg_at_k, precision_at_k, recall_at_k)

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))

class VowpalRecommender():
    TOP_K = 10
    sys.path.append('./')
    sys.path.append('./vwdata')

    model_path = 'vwdata/vw.model'
    saved_model_path = 'vwdata/vw_saved.model'
    train_path = 'vwdata/train.dat'
    test_path = 'vwdata/test.dat'
    train_logistic_path = 'vwdata/train_logistic.dat'
    test_logistic_path ='vwdata/test_logistic.dat'
    prediction_path ='vwdata/prediction.dat'
    all_test_path ='vwdata/new_test.dat'
    all_prediction_path ='vwdata/new_prediction.dat'

    def __init__(self, data, n=100000, scale=5):
        self.n = n
        self.dataset = self.build_dataset(data, scale)

        # load movielens data 
        self.df = self.dataset
        # split data to train and test sets, default values take 75% of each users ratings as train, and 25% as test
        self.train, self.test = python_random_split(self.df, 0.75)

        # save train and test data in vw format
        self.to_vw(df=self.train, output=self.train_path)
        self.to_vw(df=self.test, output=self.test_path)

        # save data for logistic regression (requires adjusting the label)
        self.to_vw(df=self.train, output=self.train_logistic_path, logistic=True)
        self.to_vw(df=self.test, output=self.test_logistic_path, logistic=True)

       
        
        
    def build_dataset(self, data, scale):
        data.rename(columns={'Llaves': 'itemID', 'cluster':'userID', 'Peso del prestamos': 'rating'}, inplace=True)
                
        data.drop('Dewey', inplace=True, axis=1)
        maxRat = data['rating'].max()
        mappedWeights = data['rating'].map(lambda x: interp(x,[0, maxRat],[0, scale]))
        data['rating'] = mappedWeights
        data['timestamp']  = data['timestamp'].map(lambda x: x.timestamp())
        data['itemID'] = data['itemID'].map(lambda x: x if x != "#XL_EVAL_ERROR#" else 0)
        data['rating'] = data['rating'].astype('int64')
        data['itemID'] = data['itemID'].astype('int64')
        data['userID'] = data['userID'].astype('int64')
        data['timestamp'] = data['timestamp'].astype('int64')

        return data

    def to_vw(self, df, output, logistic=False):
        """Convert Pandas DataFrame to vw input format
        Args:
            df (pd.DataFrame): input DataFrame
            output (str): path to output file
            logistic (bool): flag to convert label to logistic value
        """
        with open(output, 'w') as f:
            tmp = self.df.reset_index()

            # we need to reset the rating type to an integer to simplify the vw formatting
            tmp['itemID'] = tmp['itemID'].astype('int64')
            
            tmp['userID'] = tmp['userID'].astype('int64')
            
            # convert rating to binary value
            if logistic:
                tmp['rating'] = tmp['rating'].apply(lambda x: 1 if x >= 0.09 else -1)
            
            tmp['rating'] = tmp['rating'].astype('int64')
            # convert each row to VW input format (https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format)
            # [label] [tag]|[user namespace] [user id feature] |[item namespace] [movie id feature]
            # label is the true rating, tag is a unique id for the example just used to link predictions to truth
            # user and item namespaces separate the features to support interaction features through command line options
            for _, row in tmp.iterrows():
                f.write('{rating:d} {index:d}|user {userID:d} |item {itemID:d}\n'.format_map(row.astype('int')))
    
    def run_vw(self, train_params, test_params, test_data, prediction_path, logistic=False):
        """Convenience function to train, test, and show metrics of interest
        Args:
            train_params (str): vw training parameters
            test_params (str): vw testing parameters
            test_data (pd.dataFrame): test data
            prediction_path (str): path to vw prediction output
            logistic (bool): flag to convert label to logistic value
        Returns:
            (dict): metrics and timing information
        """

        # train model
        train_start = process_time()
        run(train_params.split(' '), check=True)
        train_stop = process_time()
        
        # test model
        test_start = process_time()
        run(test_params.split(' '), check=True)
        test_stop = process_time()
        
        # read in predictions
        pred_df = pd.read_csv(prediction_path, delim_whitespace=True, names=['prediction'], index_col=1).join(test_data)
        pred_df.drop("rating", axis=1, inplace=True)

        test_df = test_data.copy()
        if logistic:
            # make the true label binary so that the metrics are captured correctly
            test_df['rating'] = test['rating'].apply(lambda x: 1 if x >= 0.09 else -1)
        else:
            # ensure results are integers in correct range
            pred_df['prediction'] = pred_df['prediction'].apply(lambda x: int(max(1, min(5, round(x)))))

        # calculate metrics
        result = dict()
        pred_df.fillna(value=0, inplace=True)
        pred_df['itemID'] = pred_df['itemID'].astype('int64')
        pred_df['userID'] = pred_df['userID'].astype('int64')
        pred_df['timestamp'] = pred_df['timestamp'].astype('int64')

        result['RMSE'] = rmse(test_df, pred_df)
        result['MAE'] = mae(test_df, pred_df)
        result['R2'] = rsquared(test_df, pred_df)
        result['Explained Variance'] = exp_var(test_df, pred_df)
        result['Train Time (ms)'] = (train_stop - train_start) * 1000
        result['Test Time (ms)'] = (test_stop - test_start) * 1000
        
        return result

    def fit(self):
        with Timer() as train_time:
            self.linear_reg()
            self.linear_reg_interact()
            self.multinomial_logistic_reg()
            self.logistic_reg()
            self.svd_matrix_fact()
            self.fact_machine_matrix_fact()

            # First construct a test set of all items (except those seen during training) for each user
            users = self.df[['userID']].drop_duplicates()
            users['key'] = 1

            items = self.df[['itemID']].drop_duplicates()
            items['key'] = 1

            all_pairs = pd.merge(users, items, on='key').drop(columns=['key'])

            # now combine with training data and keep only entries that were note in training
            merged = pd.merge(train[['userID', 'itemID', 'rating']], all_pairs, on=["userID", "itemID"], how="outer")
            self.all_user_items = merged[merged['rating'].isnull()].fillna(0).astype('int64')

            # save in vw format (this can take a while)
            self.to_vw(df=self.all_user_items, output=self.all_test_path)
            

        print("Took {} seconds for training.".format(train_time.interval))

    def linear_reg(self):
        train_params = 'vw -f {model} -d {data} --quiet'.format(model=self.model_path, data=self.train_path)
        # save these results for later use during top-k analysis
        test_params = 'vw -i {model} -d {data} -t -p {pred} --quiet'.format(model=self.model_path, data=self.test_path, pred=self.prediction_path)

        result = self.run_vw(train_params=train_params, 
                        test_params=test_params, 
                        test_data=self.test, 
                        prediction_path=self.prediction_path)

        self.comparison = pd.DataFrame(result, index=['Linear Regression'])

    def linear_reg_interact(self):
        train_params = 'vw -b 26 -q ui -f {model} -d {data} --quiet'.format(model=self.saved_model_path, data=self.train_path)
        test_params = 'vw -i {model} -d {data} -t -p {pred} --quiet'.format(model=self.saved_model_path, data=self.test_path, pred=self.prediction_path)

        result = self.run_vw(train_params=train_params,
                        test_params=test_params,
                        test_data=self.test,
                        prediction_path=self.prediction_path)
        self.saved_result = result

        self.comparison = self.comparison.append(pd.DataFrame(result, index=['Linear Regression w/ Interaction']))
                
    def multinomial_logistic_reg(self):
        train_params = 'vw --loss_function logistic --oaa 5 -f {model} -d {data} --quiet'.format(model=self.model_path, data=self.train_path)
        test_params = 'vw --link logistic -i {model} -d {data} -t -p {pred} --quiet'.format(model=self.model_path, data=self.test_path, pred=self.prediction_path)

        result = self.run_vw(train_params=train_params,
                        test_params=test_params,
                        test_data=self.test,
                        prediction_path=self.prediction_path)

        self.comparison = cself.omparison.append(pd.DataFrame(result, index=['Multinomial Regression']))
    
    def logistic_reg(self):
        train_params = 'vw --loss_function logistic -f {model} -d {data} --quiet'.format(model=self.model_path, data=self.train_logistic_path)
        test_params = 'vw --link logistic -i {model} -d {data} -t -p {pred} --quiet'.format(model=self.model_path, data=self.test_logistic_path, pred=self.prediction_path)

        result = self.run_vw(train_params=train_params,
                        test_params=test_params,
                        test_data=self.test,
                        prediction_path=self.prediction_path,
                        logistic=True)

        self.comparison = self.comparison.append(pd.DataFrame(result, index=['Logistic Regression']))

    def svd_matrix_fact(self):
        train_params = 'vw --rank 5 -q ui -f {model} -d {data} --quiet'.format(model=self.model_path, data=self.train_path)
        test_params = 'vw -i {model} -d {data} -t -p {pred} --quiet'.format(model=self.model_path, data=self.test_path, pred=self.prediction_path)

        result = self.run_vw(train_params=train_params,
                        test_params=test_params,
                        test_data=self.test,
                        prediction_path=self.prediction_path)

        self.comparison = self.comparison.append(pd.DataFrame(result, index=['Matrix Factorization (Rank)']))

    def fact_machine_matrix_fact(self):
        train_params = 'vw --lrq ui7 -f {model} -d {data} --quiet'.format(model=self.model_path, data=self.train_path)
        test_params = 'vw -i {model} -d {data} -t -p {pred} --quiet'.format(model=self.model_path, data=self.test_path, pred=self.prediction_path)

        result = self.run_vw(train_params=train_params,
                        test_params=test_params,
                        test_data=self.test,
                        prediction_path=self.prediction_path)

        self.comparison = self.omparison.append(pd.DataFrame(result, index=['Matrix Factorization (LRQ)']))
                        
                
    def predict(self):
        # run the saved model (linear regression with interactions) on the new dataset
        test_start = process_time()
        test_params = 'vw -i {model} -d {data} -t -p {pred} --quiet'.format(model=self.saved_model_path, data=self.all_test_path, pred=self.prediction_path)
        run(test_params.split(' '), check=True)
        test_stop = process_time()
        test_time = test_stop - test_start

        # load predictions and get top-k from previous saved results
        pred_data = pd.read_csv(self.prediction_path, delim_whitespace=True, names=['prediction'], index_col=1).join(self.all_user_items)
        self.top_k = get_top_k_items(pred_data, col_rating='prediction', k=self.TOP_K)[['prediction', 'userID', 'itemID']]


    def evaluate_model():

        # get ranking metrics
        args = [self.test, top_k]
        kwargs = dict(col_user='userID', col_item='itemID', col_rating='rating', col_prediction='prediction',
                    relevancy_method='top_k', k=self.TOP_K)

        self.rank_metrics = {'MAP': map_at_k(*args, **kwargs), 
                        'NDCG': ndcg_at_k(*args, **kwargs),
                        'Precision': precision_at_k(*args, **kwargs),
                        'Recall': recall_at_k(*args, **kwargs)}

        # final results
        all_results = ['{k}: {v}'.format(k=k, v=v) for k, v in self.saved_result.items()]
        all_results += ['{k}: {v}'.format(k=k, v=v) for k, v in self.rank_metrics.items()]
        print('\n'.join(all_results))

    def predict_for_cluster(self, cluster):
        return self.top_k.loc[top_k['userID'] == cluster]['itemID'].values


    def export_model(self):
        now_time = time.strftime("%m%d%H%m")
        filepath = "../Models/vowpal_trained_model_"+str(self.n)+"_"+now_time+".pkl"
        with gzip.open(filepath, "wb") as f:
            pickled = pickle.dumps(self.model, protocol=4)
            optimized_pickle = pickletools.optimize(pickled)
            f.write(optimized_pickle)
