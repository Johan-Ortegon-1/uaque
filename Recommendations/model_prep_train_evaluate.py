
import pandas as pd
import numpy as np

# Archivo donde se calcularon los pesos para cada prestamo

# COLAB
# df_prestamos = pd.read_json('/content/drive/MyDrive/UAQUE/TABLA_JOIN.json')


"""# Evaluación

## Definición de métricas
"""

import itertools

from surprise import accuracy
from collections import defaultdict

class RecommenderMetrics:

    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)

    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)

    def GetTopN(predictions, n=10, minimumRating=4.0):
        topN = defaultdict(list)


        for userID, movieID, actualRating, estimatedRating, _ in predictions:
            if (estimatedRating >= minimumRating):
                topN[(userID)].append(((movieID), estimatedRating))

        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[(userID)] = ratings[:n]

        return topN

    def HitRate(topNPredicted, leftOutPredictions):
        hits = 0
        total = 0

        # For each left-out rating
        for leftOut in leftOutPredictions:
            userID = leftOut[0]
            leftOutMovieID = leftOut[1]
            # Is it in the predicted top 10 for this user?
            hit = False
            for movieID, predictedRating in topNPredicted[userID]:
                if (leftOutMovieID == movieID):
                    hit = True
                    break
            if (hit) :
                hits += 1

            total += 1

        # Compute overall precision
        return hits/total

    def CumulativeHitRate(topNPredicted, leftOutPredictions, ratingCutoff=0):
        hits = 0
        total = 0

        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Only look at ability to recommend things the users actually liked...
            if (actualRating >= ratingCutoff):
                # Is it in the predicted top 10 for this user?
                hit = False
                for movieID, predictedRating in topNPredicted[userID]:
                    if ((leftOutMovieID) == movieID):
                        hit = True
                        break
                if (hit) :
                    hits += 1

                total += 1

        # Compute overall precision
        return hits/total

    def RatingHitRate(topNPredicted, leftOutPredictions):
        hits = defaultdict(float)
        total = defaultdict(float)

        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hit = False
            for movieID, predictedRating in topNPredicted[(userID)]:
                if ((leftOutMovieID) == movieID):
                    hit = True
                    break
            if (hit) :
                hits[actualRating] += 1

            total[actualRating] += 1

        # Compute overall precision
        for rating in sorted(hits.keys()):
            print (rating, hits[rating] / total[rating])

    def AverageReciprocalHitRank(topNPredicted, leftOutPredictions):
        summation = 0
        total = 0
        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hitRank = 0
            rank = 0
            for movieID, predictedRating in topNPredicted[(userID)]:
                rank = rank + 1
                if ((leftOutMovieID) == movieID):
                    hitRank = rank
                    break
            if (hitRank > 0) :
                summation += 1.0 / hitRank

            total += 1

        return summation / total

    # What percentage of users have at least one "good" recommendation
    def UserCoverage(topNPredicted, numUsers, ratingThreshold=0):
        hits = 0
        for userID in topNPredicted.keys():
            hit = False
            for movieID, predictedRating in topNPredicted[userID]:
                if (predictedRating >= ratingThreshold):
                    hit = True
                    break
            if (hit):
                hits += 1

        return hits / numUsers

    def Diversity(topNPredicted, matrix, simsAlgo):
        n = 1
        total = 1
        simsMatrix = matrix
        for userID in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[userID], 2)
            for pair in pairs:
                movie1 = pair[0][0]
                movie2 = pair[1][0]
                print("Item 1 ", movie1, "Item 2 ", movie2)
                try:
                    innerID1 = simsAlgo.trainset.to_inner_iid(str(movie1))
                    innerID2 = simsAlgo.trainset.to_inner_iid(str(movie2))
                    similarity = simsMatrix[innerID1][innerID2]
                    total += similarity
                except ValueError:
                     print(" ") 
                n += 1

        S = total / n
        return (1-S)

    def Novelty(topNPredicted, rankings):
        n = 1
        total = 1
        for userID in topNPredicted.keys():
            for rating in topNPredicted[userID]:
                movieID = rating[0]
                rank = rankings[movieID]
                total += rank
                n += 1
        return total / n

"""## Evaluación de un modelo"""

class EvaluatedAlgorithm:
    
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name
        
    def Evaluate(self, evaluationData, doTopN, n=10, verbose=True):
        metrics = {}
        # Calculando Precisión
        if (verbose):
            print("Calculando Precisión...")
        self.algorithm.fit(evaluationData.GetTrainSet())
        predictions = self.algorithm.test(evaluationData.GetTestSet())
        metrics["RMSE"] = RecommenderMetrics.RMSE(predictions)
        metrics["MAE"] = RecommenderMetrics.MAE(predictions)
        
        if (doTopN):
            # Evaluando los top-10 utilizando Leave One Out
            if (verbose):
                print("Evaluando los top-10 utilizando Leave One Out...")

            self.algorithm.fit(evaluationData.GetLOOCVTrainSet())
            leftOutPredictions = self.algorithm.test(evaluationData.GetLOOCVTestSet())        

            #Construir predicciones para todos los ratings que no están en el set de entrenamiento
            allPredictions = self.algorithm.test(evaluationData.GetLOOCVAntiTestSet())

            # Calula las top 10 recomendaciones para cada usuario
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)

            if (verbose):
                print("Computing hit-rate and rank metrics...")
            # Para ver qué tan seguido se recomienda un item que el usuario haya prestado
            metrics["HR"] = RecommenderMetrics.HitRate(topNPredicted, leftOutPredictions)  
            # Para ver qué tan seguido se recomienda un ítem que al usuario le haya gustado
            metrics["cHR"] = RecommenderMetrics.CumulativeHitRate(topNPredicted, leftOutPredictions)
            # Calcular ARHR
            metrics["ARHR"] = RecommenderMetrics.AverageReciprocalHitRank(topNPredicted, leftOutPredictions)
        
            #Evaluate properties of recommendations on full training set
            #Evaluando propiedades de las recomendaciones en el set completo de entrenamiento
            if (verbose):
                print("Calculando recomiendaciones para todo el data set...")
            self.algorithm.fit(evaluationData.GetFullTrainSet())
            allPredictions = self.algorithm.test(evaluationData.GetFullAntiTestSet())
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)
            if (verbose):
                print("Analizando cobertura, diversidad y novedad...")

            # Imprime cobertura para un usuario con una predicción de mínimo 4.0
            # OJO QUE NUESTRA ESCALA NO ESTÁ DE 0 a 5
            # CONSIDERAR HACER UN MAP AL PESO O CAMBIAR EL THRESHOLD
            metrics["Coverage"] = RecommenderMetrics.UserCoverage(  topNPredicted, 
                                                                   evaluationData.GetFullTrainSet().n_users, 
                                                                   ratingThreshold=4.0)
            # Measure diversity of recommendations:
            metrics["Diversity"] = RecommenderMetrics.Diversity(topNPredicted, evaluationData.GetSimMatrix(), evaluationData.GetSimilarities())
            
            # Measure novelty (average popularity rank of recommendations):
            metrics["Novelty"] = RecommenderMetrics.Novelty(topNPredicted, 
                                                            evaluationData.GetPopularityRankings())
        
        if (verbose):
            print("Análisis Completado")
    
        return metrics
    
    def GetName(self):
        return self.name
    
    def GetAlgorithm(self):
        return self.algorithm

"""### Evaluador"""

# Me permite añadir los modelos que voy a evaluar para luego hacer las comparaciones
class Evaluator:
    
    algorithms = []
    dataset = {}
    
    def __init__(self, dataset, rankings):
        ed = EvaluationData(dataset, rankings)
        self.dataset = ed
        
    def AddAlgorithm(self, algorithm, name):
        newAlg = []
        alg = EvaluatedAlgorithm(algorithm, name)
        newAlg.append(alg)
        self.algorithms = newAlg
        
    def Evaluate(self, doTopN):
        results = {}
        for algorithm in self.algorithms:
            print("Evaluando el algoritmo ", algorithm.GetName(), "...")
            results[algorithm.GetName()] = algorithm.Evaluate(self.dataset, doTopN)

        # Print results
        print("\n")
        
        if (doTopN):
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                    "Algoritmo", "RMSE", "MAE", "HR", "cHR", "ARHR", "Cobertura", "Diversidad", "Novedad"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                        name, metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["cHR"], metrics["ARHR"],
                                      metrics["Coverage"], metrics["Diversity"], metrics["Novelty"]))
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"]))
                
        print("\nResumen:\n")
        print("RMSE:      Root Mean Squared Error. Valores más bajos significa mejor precisión.")
        print("MAE:       Mean Absolute Error. Valores más bajos significa mejor precisión.")
        if (doTopN):
            print("HR:        Hit Rate; Qué tan seguido recomendamos una predicción Left-Out (fuera del set de entrenamiento). Más alto es mejor.")
            print("cHR:       Cumulative Hit Rate; Hit Rate limitado a las calificaciones por encima de un determinado umbral.  Más alto es mejor.")
            print("ARHR:      Average Reciprocal Hit Rank - Hit rate que tiene el ranking del ítem. Más alto es mejor." )
            print("Cobertura:  Ratio de usuarios para los que existen recomendaciones por encima de un determinado umbral. Más alto es mejor.")
            print("Diversidad: 1-S, donde S es la similaridad Promedio entre cada posible par de recomendaciones")
            print("           para un usuario. Más alto significa más diverso.")
            print("Novedad:   Popularidad promedio en el raking de ítems recomendados. Más alto significa más novedoso.")



"""## Preparación de los datos a evaluar"""

from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline

class EvaluationData:
    
    def __init__(self, data, popularityRankings):
        
        self.rankings = popularityRankings
        
        #Se construye un set completo de entrenamiento para evaluar
        self.fullTrainSet = data.build_full_trainset()
        self.fullAntiTestSet = self.fullTrainSet.build_anti_testset()
        
        #Se construye un set de entrenamiento 75/25 para medir la precisión
        self.trainSet, self.testSet = train_test_split(data, test_size=.25, random_state=1)
        
        #Construir un set de entrenamiento de "dejar uno fuera LOO" para evaluar los recomendadores top-N
        #Y construir un "anti-test-set" para construir predicciones
        LOOCV = LeaveOneOut(n_splits=1, random_state=1)
        for train, test in LOOCV.split(data):
            self.LOOCVTrain = train
            self.LOOCVTest = test
            
        self.LOOCVAntiTestSet = self.LOOCVTrain.build_anti_testset()
        
        #Calcula la matriz de similaridad entre items para medir diversidad
        sim_options = {'name': 'cosine', 'user_based': False}
        self.simsAlgo = KNNBaseline(sim_options=sim_options)
        self.simsAlgo.fit(self.fullTrainSet)
            
    def GetFullTrainSet(self):
        return self.fullTrainSet
    
    def GetFullAntiTestSet(self):
        return self.fullAntiTestSet
    
    def GetAntiTestSetForUser(self, testSubject):
        trainset = self.fullTrainSet
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid((testSubject))
        user_items = set([j for (j, _) in trainset.ur[u]])
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                                 i in trainset.all_items() if
                                 i not in user_items]
        return anti_testset

    def GetTrainSet(self):
        return self.trainSet
    
    def GetTestSet(self):
        return self.testSet
    
    def GetLOOCVTrainSet(self):
        return self.LOOCVTrain
    
    def GetLOOCVTestSet(self):
        return self.LOOCVTest
    
    def GetLOOCVAntiTestSet(self):
        return self.LOOCVAntiTestSet
    
    def GetSimilarities(self):
        return self.simsAlgo

    def GetSimMatrix(self):
        return self.simsAlgo.sim

    def GetPopularityRankings(self):
        return self.rankings

