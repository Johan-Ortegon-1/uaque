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
import surprise
import papermill as pm
import scrapbook as sb
import pandas as pd
from surprise import Reader
from numpy import interp

from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k, precision_at_k, 
                                                     recall_at_k, get_top_k_items)
from recommenders.models.surprise.surprise_utils import predict, compute_ranking_predictions

print("System version: {}".format(sys.version))
print("Surprise version: {}".format(surprise.__version__))


class SVDRecommender():
    TOP_K = 10

    def __init__(self, data, n=100000, scale=5):
        self.n = n
        self.dataset = self.build_dataset(data, scale)

        self.train, self.test = python_random_split(self.dataset, 0.75)

        reader = Reader(rating_scale=(0, scale))
        self.train_set = surprise.Dataset.load_from_df(self.train, reader=reader).build_full_trainset()

        self.model = surprise.SVD(random_state=0, n_factors=200, n_epochs=30, verbose=True)
        
        
    def build_dataset(self, data, scale):
        data.rename(columns={'Llaves': 'itemID', 'cluster':'userID', 'Peso del prestamos': 'rating'}, inplace=True)
        maxRat = data['rating'].max()
        mappedWeights = data['rating'].map(lambda x: interp(x,[0, maxRat],[0, scale]))
        data['rating'] = mappedWeights
#         data.drop('Dewey', inplace=True, axis=1)
        data.drop('timestamp', inplace=True, axis=1)
        return data

    def fit(self):
        with Timer() as train_time:
            self.model.fit(self.train_set)

        print("Took {} seconds for training.".format(train_time.interval))
        
        
    def predict(self):
        self.predictions = predict(self.model, self.test, usercol='userID', itemcol='itemID')
        with Timer() as test_time:
            self.all_predictions = compute_ranking_predictions(self.model, self.train, usercol='userID', itemcol='itemID', remove_seen=True)
            
        print("Took {} seconds for prediction.".format(test_time.interval))


    def evaluate_model():

        test = self.test
        predictions = self.predictions
        all_predictions = self.all_predictions
        
        eval_rmse = rmse(test, predictions)
        eval_mae = mae(test, predictions)
        eval_rsquared = rsquared(test, predictions)
        eval_exp_var = exp_var(test, predictions)

        k = 10
        eval_map = map_at_k(test, all_predictions, col_prediction='prediction', k=k)
        eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=k)
        eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=k)
        eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=k)


        print("RMSE:\t\t%f" % eval_rmse,
            "MAE:\t\t%f" % eval_mae,
            "rsquared:\t%f" % eval_rsquared,
            "exp var:\t%f" % eval_exp_var, sep='\n')

        print('----')

        print("MAP:\t%f" % eval_map,
            "NDCG:\t%f" % eval_ndcg,
            "Precision@K:\t%f" % eval_precision,
            "Recall@K:\t%f" % eval_recall, sep='\n')

    def predict_for_cluster(self, cluster):
        preds_cluster = self.all_predictions.loc[self.all_predictions['userID'] == cluster]
        preds_cluster.sort_values('prediction')
        top_10_preds = preds_cluster[:10]
    
        return top_10_preds['itemID'].values


    def export_model(self):
        now_time = time.strftime("%m%d%H%m")
        filepath = "../Models/svd_trained_model_"+str(self.n)+"_"+now_time+".pkl"
        with gzip.open(filepath, "wb") as f:
            pickled = pickle.dumps(self.model, protocol=4)
            optimized_pickle = pickletools.optimize(pickled)
            f.write(optimized_pickle)

