from surprise import AlgoBase
from surprise import PredictionImpossible
import math
import numpy as np
import heapq
import sys
sys.path.append('./')
import model_prep_train_evaluate as cbrec


class ContentKNNAlgorithm(AlgoBase):

    
    def __init__(self, simMatrix, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k
        self.simMatrix = simMatrix

    def fit(self, trainset):
        print("Pre fit base")
        AlgoBase.fit(self, trainset)
        print("Post fit base")
        # Calculando la matrix de similaridad entre ítems basado en su contenido
        # En este caso solo genres = temáticas y years = Año de publicación
        # genres = getAllThemes()
        # years = getAllPublishedYears()

        print("Calculando matriz de similaridad basada en contenido...")
        # Calcula la distancia entre Dewys para cada combinación de ítems como una matriz 2x2
        simsalgo = self.simMatrix
        self.similarities = simsalgo

        #         for thisRating in range(self.trainset.n_items):
        #             if (thisRating % 100 == 0):
        #                 print(thisRating, " de ", self.trainset.n_items)
        #             for otherRating in range(thisRating+1, self.trainset.n_items):
        #                 thisItemID = int(self.trainset.to_raw_iid(thisRating))
        #                 print("thisItemID", thisItemID)
        #                 otherItemID = int(self.trainset.to_raw_iid(otherRating))
        #                 print("otherItemID", otherItemID)
        #                 genreSimilarity = self.computeGenreSimilarity(thisItemID, otherItemID, genres)
        #                 yearSimilarity = self.computeYearSimilarity(thisItemID, otherItemID, years)
        #                 self.similarities[thisRating, otherRating] = genreSimilarity * yearSimilarity
        #                 self.similarities[otherRating, thisRating] = self.similarities[thisRating, otherRating]

        print("...done.")

        return self

    # Calculo de similitud de temáticas

    # Calculo de similitud de año de publicación
    def computeYearSimilarity(self, item1, item2, years):
        diff = abs(years[str(float(item1))] - years[str(float(item2))])
        sim = math.exp(-diff / 10.0)
        return sim

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('Usario o item desconocidos')

        # Construimos los puntajes entre el ítem y las pesos dados a los temas
        neighbors = []
        for rating in self.trainset.ur[u]:
            genreSimilarity = self.similarities[i, rating[0]]
            neighbors.append((genreSimilarity, rating[1]))

        # Extraer los top-K ratings más similares
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        # Calculo promedio de similitud de los K vecinos con respecto a los ratings del usuario
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if (simScore > 0):
                simTotal += simScore
                weightedSum += simScore * rating

        if (simTotal == 0):
            raise PredictionImpossible('No tiene vecinos')

        predictedRating = weightedSum / simTotal

        return predictedRating

