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
from recommenders.models.sar import SAR
import sys
import gzip, pickle, pickletools

class SarRecommeder():
    TOP_K = 10

    def __init__(self, data, n=100000, scale=5):
        self.n = n
        self.dataset = self.build_dataset(data, scale)
        self.train, self.test = python_stratified_split(self.dataset, ratio=0.75, col_user='userID', col_item='itemID', seed=42)

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(levelname)-8s %(message)s')


        self.model = SAR(
            col_user="userID",
            col_item="itemID",
            col_rating="rating",
            col_timestamp="timestamp",
            similarity_type="jaccard",
            time_decay_coefficient=30,
            timedecay_formula=True,
            normalize=True
        )

    def build_dataset(self, data, scale):
        dfP = data[0:self.n]
        scale = scale

        toSurprisedb = pd.DataFrame(columns=['userID', 'itemID', 'rating'])

        toSurprisedb['userID'] = dfP['cluster']

        toSurprisedb['itemID'] = dfP['Llaves']

        maxRat = dfP['Peso del prestamos'].max()
        mappedWeights = dfP['Peso del prestamos'].map(lambda x: interp(x, [0, maxRat], [0, scale]))
        toSurprisedb['rating'] = mappedWeights
        toSurprisedb['timestamp'] = dfP['timestamp'].map(lambda x: x.timestamp())

        return toSurprisedb

    def fit(self):
        self.model.fit(self.train)
        
    def predict(self):
        self.model.recommend_k_items(self.test, remove_seen = False)

    def evaluate_model(self, top_k):
        eval_map = map_at_k(self.test, top_k, col_user='userID', col_item='itemID', col_rating='rating', k=self.TOP_K)
        eval_ndcg = ndcg_at_k(self.test, top_k, col_user='userID', col_item='itemID', col_rating='rating', k=self.TOP_K)
        eval_precision = precision_at_k(self.test, top_k, col_user='userID', col_item='itemID', col_rating='rating', k=self.TOP_K)
        eval_recall = recall_at_k(self.test, top_k, col_user='userID', col_item='itemID', col_rating='rating', k=self.TOP_K)
        eval_rmse = rmse(self.test, top_k, col_user='userID', col_item='itemID', col_rating='rating')
        eval_mae = mae(self.test, top_k, col_user='userID', col_item='itemID', col_rating='rating')
        eval_rsquared = rsquared(self.test, top_k, col_user='userID', col_item='itemID', col_rating='rating')
        eval_exp_var = exp_var(self.test, top_k, col_user='userID', col_item='itemID', col_rating='rating')

        positivity_threshold = 2
        test_bin = self.test.copy()
        test_bin['rating'] = binarize(test_bin['rating'], positivity_threshold)

        top_k_prob = top_k.copy()
        top_k_prob['prediction'] = minmax_scale(
            top_k_prob['prediction'].astype(float)
        )

        eval_logloss = logloss(test_bin, top_k_prob, col_user='userID', col_item='itemID', col_rating='rating')

        print("Model:\t",
              "Top K:\t%d" % self.TOP_K,
              "MAP:\t%f" % eval_map,
              "NDCG:\t%f" % eval_ndcg,
              "Precision@K:\t%f" % eval_precision,
              "Recall@K:\t%f" % eval_recall,
              "RMSE:\t%f" % eval_rmse,
              "MAE:\t%f" % eval_mae,
              "R2:\t%f" % eval_rsquared,
              "Exp var:\t%f" % eval_exp_var,
              "Logloss:\t%f" % eval_logloss,
              sep='\n')

    def predict_for_cluster(self, cluster):
        ground_truth = self.test[self.test['userID'] == cluster].sort_values(by='rating', ascending=False)[:self.TOP_K]
        return ground_truth['itemID']


    def export_model(self):
        now_time = time.strftime("%m%d%H%m")
        filepath = "../Models/sar_trained_model_"+str(self.n)+"_"+now_time+".pkl"
        with gzip.open(filepath, "wb") as f:
            pickled = pickle.dumps(self.model, protocol=4)
            optimized_pickle = pickletools.optimize(pickled)
            f.write(optimized_pickle)
