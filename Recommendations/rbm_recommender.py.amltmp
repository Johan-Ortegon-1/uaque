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

# set the environment path to find Recommenders
import sys

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scrapbook as sb
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.models.rbm.rbm import RBM
from recommenders.datasets.python_splitters import numpy_stratified_split
from recommenders.datasets.sparse import AffinityMatrix


from recommenders.datasets import movielens
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k

class RBMRecommender():
    TOP_K = 10

    def __init__(self, data, n=100000, scale=5):
        self.n = n
        self.dataset = self.build_dataset(data, scale)

        #to use standard names across the analysis 
        header = {
                "col_user": "userID",
                "col_item": "movieID",
                "col_rating": "rating",
            }

        #instantiate the sparse matrix generation  
        self.am = AffinityMatrix(df = data, **header)

        #obtain the sparse matrix 
        X, _, _ = self.am.gen_affinity_matrix()
        self.Xtr, self.Xtst = numpy_stratified_split(X)
        self.model = RBM(hidden_units= 600, training_epoch = 30, minibatch_size= 60, keep_prob=0.9,with_metrics =True)
        
        
    def build_dataset(self, data, scale):
        data.rename(columns={'Llaves': 'movieID', 'cluster':'userID', 'Peso del prestamos': 'rating'}, inplace=True)
        scale = 5
        maxRat = data['rating'].max()
        mappedWeights = data['rating'].map(lambda x: interp(x,[0, maxRat],[1, scale*2]))
        data['rating'] = mappedWeights
        data.loc[:, 'rating'] = data['rating'].astype(np.int32) 
        return data

    def fit(self):
        self.traintime = self.model.fit(self.Xtr, self.Xtst)
        
        
    def predict(self):
        #number of top score elements to be recommended  
        K = 10

        #Model prediction on the test set Xtst. 
        self.top_k, self.testtime =  self.model.recommend_k_items(self.Xtst)
        self.top_k_df = self.am.map_back_sparse(self.top_k, kind = 'prediction')
        self.test_df = self.am.map_back_sparse(self.Xtst, kind = 'ratings')

    def evaluate_model(
        data_size,
        data_true,
        data_pred,
        K
    ):

        time_train = self.traintime
        time_test = self.testtime

        eval_map = map_at_k(data_true, data_pred, col_user="userID", col_item="movieID", 
                        col_rating="rating", col_prediction="prediction", 
                        relevancy_method="top_k", k= K)

        eval_ndcg = ndcg_at_k(data_true, data_pred, col_user="userID", col_item="movieID", 
                        col_rating="rating", col_prediction="prediction", 
                        relevancy_method="top_k", k= K)

        eval_precision = precision_at_k(data_true, data_pred, col_user="userID", col_item="movieID", 
                                col_rating="rating", col_prediction="prediction", 
                                relevancy_method="top_k", k= K)

        eval_recall = recall_at_k(data_true, data_pred, col_user="userID", col_item="movieID", 
                            col_rating="rating", col_prediction="prediction", 
                            relevancy_method="top_k", k= K)

        
        df_result = pd.DataFrame(
            {   "Dataset": data_size,
                "K": K,
                "MAP": eval_map,
                "nDCG@k": eval_ndcg,
                "Precision@k": eval_precision,
                "Recall@k": eval_recall,
                "Train time (s)": time_train,
                "Test time (s)": time_test
            }, 
            index=[0]
        )
        
        return df_result

    def predict_for_cluster(self, cluster):
        return self.top_k_df.loc[self.top_k_df['userID'] == cluster]['movieID'].astype(np.int32).values


    def export_model(self):
        now_time = time.strftime("%m%d%H%m")
        filepath = "../Models/rbm_trained_model_"+str(self.n)+"_"+now_time+".pkl"
        with gzip.open(filepath, "wb") as f:
            pickled = pickle.dumps(self.model, protocol=4)
            optimized_pickle = pickletools.optimize(pickled)
            f.write(optimized_pickle)