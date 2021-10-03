from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
import random
import numpy as np
from numpy import interp
from sklearn.utils import shuffle
import pandas as pd
from content_knn_algorithm import ContentKNNAlgorithm
import sys
sys.path.append('./')
import model_prep_train_evaluate as cbrec
import gzip, pickle, pickletools
import time
import json
from collections import defaultdict
from surprise import AlgoBase
from surprise import PredictionImpossible
import math
import heapq

class ContentKNNRecommender():
    def __init__(self, data, n=10000, scale=5):
        self.n = n
        self.dataset = self.build_dataset(data)
        reader = Reader(rating_scale=(0, scale))
        evaluate = Dataset.load_from_df(self.dataset[['userID', 'itemID', 'rating']], reader)
        rankings = self.getPopularityRank(data)
        self.evaluator = cbrec.Evaluator(evaluate, rankings)
        contentKNN = ContentKNNAlgorithm(simMatrix=self.evaluator.dataset.GetSimMatrix())
        self.evaluator.AddAlgorithm(contentKNN, "ContentKNN")
        self.train = self.evaluator.dataset.trainSet


    def build_dataset(self, data):
        dfP = (data[0:self.n])
        # dfP = df
        scale = 5

        toSurprisedb = pd.DataFrame(columns=['userID', 'itemID', 'rating'])

        # Para USUARIOS
        # toSurprisedb['userID'] = dfP['ID de usuario ok']

        # Para CLUSTER
        toSurprisedb['userID'] = dfP['cluster']

        toSurprisedb['itemID'] = dfP['Llaves']

        maxRat = dfP['Peso del prestamos'].max()
        mappedWeights = dfP['Peso del prestamos'].map(lambda x: interp(x, [0, maxRat], [0, scale]))
        toSurprisedb['rating'] = mappedWeights

        return toSurprisedb

    def fit_evaluate(self, verbose=False):
        self.evaluator.Evaluate(verbose)

    # Función para obtener los items ordenados según su rango, este depende del peso dado a cada ítem y de cuántas veces ha sido calificado
    def getPopularityRank(self, data):
        df = data
        ratings = defaultdict(int)
        rankings = defaultdict(int)

        for llave in df['Llaves']:
            ratings[llave] += 1

        rank = 1
        for llave, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[llave] = rank
            rank += 1
        return rankings

    def GetAntiTestSetForUser(self, testSubject):
        trainset = self.train
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid((testSubject))
        user_items = set([j for (j, _) in trainset.ur[u]])
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                         i in trainset.all_items() if
                         i not in user_items]
        return anti_testset

    def predict_for_cluster(self, cluster, k=10):
        testSubject = cluster
        print(len(list(set(self.evaluator.algorithms))))

        for algo in set(self.evaluator.algorithms):
            print("\nUsando el recomendador: ", algo.GetName())

            # print("\nConstruyendo el modelo de recomendación...")
            # # trainSet = self.dataset.GetFullTrainSet()
            # algo.GetAlgorithm().fit(trainSet)

            print("Computing recommendations...")
            testSet = self.GetAntiTestSetForUser(testSubject)

            predictions = algo.GetAlgorithm().test(testSet)

            recommendations = []

            for userID, movieID, actualRating, estimatedRating, _ in predictions:
                intMovieID = int(movieID)
                recommendations.append((intMovieID, estimatedRating))

            recommendations.sort(key=lambda x: x[1], reverse=True)

            data_recs = []
            for ratings in recommendations[:10]:
                data_recs.append(ratings[0])
                # data_recs.append({'Titulo': itemIdToMaterial(ratings[0]),
                #                   'Rating': ratings[1],
                #                   'Dewey': itemIdToDewey(ratings[0]),
                #                   'Tematicas': itemIdToThemes(ratings[0])})

            return data_recs

    def export_model(self):
        now_time = time.strftime("%m%d%H%m")
        filepath = "../Models/content_knn_trained_model_"+str(self.n)+"_"+now_time+".pkl"
        with gzip.open(filepath, "wb") as f:
            pickled = pickle.dumps(self.model, protocol=4)
            optimized_pickle = pickletools.optimize(pickled)
            f.write(optimized_pickle)
            
    
#     class ContentKNNAlgorithm(AlgoBase):

#         def __init__(self, k=40, sim_options={}, evaluator):
#             AlgoBase.__init__(self)
#             self.k = k
#             self.evaluator = evaluator

#         def fit(self, trainset):
#             print("Pre fit base")
#             AlgoBase.fit(self, trainset)
#             print("Post fit base")
#             # Calculando la matrix de similaridad entre ítems basado en su contenido
#             # En este caso solo genres = temáticas y years = Año de publicación
#             # genres = getAllThemes()
#             # years = getAllPublishedYears()

#             print("Calculando matriz de similaridad basada en contenido...")
#             # Calcula la distancia entre Dewys para cada combinación de ítems como una matriz 2x2
#             simsalgo = self.evaluator.dataset.GetSimMatrix()
#             self.similarities = simsalgo

#             #         for thisRating in range(self.trainset.n_items):
#             #             if (thisRating % 100 == 0):
#             #                 print(thisRating, " de ", self.trainset.n_items)
#             #             for otherRating in range(thisRating+1, self.trainset.n_items):
#             #                 thisItemID = int(self.trainset.to_raw_iid(thisRating))
#             #                 print("thisItemID", thisItemID)
#             #                 otherItemID = int(self.trainset.to_raw_iid(otherRating))
#             #                 print("otherItemID", otherItemID)
#             #                 genreSimilarity = self.computeGenreSimilarity(thisItemID, otherItemID, genres)
#             #                 yearSimilarity = self.computeYearSimilarity(thisItemID, otherItemID, years)
#             #                 self.similarities[thisRating, otherRating] = genreSimilarity * yearSimilarity
#             #                 self.similarities[otherRating, thisRating] = self.similarities[thisRating, otherRating]

#             print("...done.")

#             return self

#         # Calculo de similitud de temáticas

#         # Calculo de similitud de año de publicación
#         def computeYearSimilarity(self, item1, item2, years):
#             diff = abs(years[str(float(item1))] - years[str(float(item2))])
#             sim = math.exp(-diff / 10.0)
#             return sim

#         def estimate(self, u, i):

#             if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
#                 raise PredictionImpossible('Usario o item desconocidos')

#             # Construimos los puntajes entre el ítem y las pesos dados a los temas
#             neighbors = []
#             for rating in self.trainset.ur[u]:
#                 genreSimilarity = self.similarities[i, rating[0]]
#                 neighbors.append((genreSimilarity, rating[1]))

#             # Extraer los top-K ratings más similares
#             k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

#             # Calculo promedio de similitud de los K vecinos con respecto a los ratings del usuario
#             simTotal = weightedSum = 0
#             for (simScore, rating) in k_neighbors:
#                 if (simScore > 0):
#                     simTotal += simScore
#                     weightedSum += simScore * rating

#             if (simTotal == 0):
#                 raise PredictionImpossible('No tiene vecinos')

#             predictedRating = weightedSum / simTotal

#             return predictedRating