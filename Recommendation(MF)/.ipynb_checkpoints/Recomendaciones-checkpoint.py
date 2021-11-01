import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
import datetime
import math
import cloudpickle as cp
import urllib.request
import gzip, pickle, pickletools
from PerfilGrupal import PerfilGrupal 
class Recomendaciones:
    
    def __init__(self):
        print("inicializando")
        self.cargaDatos()
        pd.options.mode.chained_assignment = None
    
    def cargaDatos(self):
        #importamos la tabla de join 
        self.join = pd.read_json('https://www.dropbox.com/s/i1komhf7u1c4y95/joinTablas.json?dl=1')
        self.pesos_clustering_unidad = pd.read_json('https://www.dropbox.com/s/6j30n8y3fn8358l/pesos_clustering_unidad.json?dl=1')
        self.pesos_clustering_decena = pd.read_json('https://www.dropbox.com/s/6m7vbpfq8b8qz4s/pesos_clustering_decena.json?dl=1')
        self.pesos_clustering_centena = pd.read_json('https://www.dropbox.com/s/3rjqco5swu55cna/pesos_clustering_centena.json?dl=1')
        self.pesos_usuarios_unidad = pd.read_json('https://www.dropbox.com/s/aitygqwn9q47rlg/pesos_usuario_x_dewey_unidad.json?dl=1')
        self.pesos_usuarios_decena = pd.read_json('https://www.dropbox.com/s/vr6ehn8xjhojuba/pesos_usuario_x_dewey_decena.json?dl=1')
        self.pesos_usuarios_centena = pd.read_json('https://www.dropbox.com/s/2vnntgjnqpijkgg/pesos_usuario_x_dewey_centena.json?dl=1')
        #Eliminamos algunas columnas que no nos interesan para este notebook
        self.join = self.join.drop(["Fecha","Dewey","Facultad","Temas","Union","TipoItem"], axis=1)
        
    def cargarModelo(self):
        tree_model_path = "https://www.dropbox.com/s/2u4kgwtbpbejfvo/Tree_trained_model_10292010.pkl?dl=1"
        with gzip.open(urllib.request.urlopen(tree_model_path), 'rb') as f:
            p = pickle.Unpickler(f)
            self.tree_predictor = p.load()
        
    def crearPesos(self):
        now = datetime.datetime.now()
        anio_actual = int(now.year)
        #Creamos la columna pesos para el dataframe a partir del año en que se realizó el prestamo
        self.join["Peso"] = self.join.apply(lambda row: 1/2**(anio_actual-row.Year), axis=1 )
        self.join[["Year","Peso"]]
            
    #Elimina los valores diferentes de cero para una fila especifica
    #Sirve para visualizar únicamente los deweys donde el usuario tiene prestamos. 
    def eliminar_cero(self,id_usuario, df_pesos):
        m1 = (df_pesos['IDUsuario'] == id_usuario)
        m2 = (df_pesos[m1] != 0).all()
        return df_pesos.loc[m1,m2]
    
    #Parámetros:
     #cluster_pertenencia:dataframe con la pertenencia de los usuarios a un cluster.
     #cluster: cluster a evaluar
     #columna: define la columna sobre la cual se va a realizar el join 
    #{DeweyUnidad|DeweyDecena|DeweyCentena}
    def rank_material(self,pesos_clustering, cluster, columna):
        #Buscamos únicamente a los usuarios que tienen un grado de pertenencia al cluster.
        usuarios = pesos_clustering.loc[pesos_clustering.Cluster == cluster]
        usuarios["Peso"] = usuarios["Peso"].apply(lambda x: float(x))
        #si el peso de algún usuario es mayor a 10 este se deja con el valor de 10
        usuarios.loc[usuarios.Peso > 10, "Peso"] = 10

        #normalizar columna de peso
        max_value = usuarios["Peso"].max()
        min_value = usuarios["Peso"].min()
        usuarios["Peso"] = (usuarios["Peso"]) / (max_value)
        #filtramos de la tabla de join los rows con el dewey
        prestamos = self.join.loc[self.join[columna] == int(cluster)]
        #display(prestamos)
        #join
        prestamos_data = pd.DataFrame(data=prestamos)
        usuario_data = pd.DataFrame(data=usuarios)
        join_tablas = pd.merge(prestamos_data, usuario_data,
                               left_on='IDUsuario',
                               right_on='IDUsuario',
                               how='inner')
        #finalmente agrupamos al material por llave y hacemos la sumatoria de sus pesos
        materialRank = join_tablas.groupby(["Llave"])["Peso_y"].sum().reset_index(name="Peso")#Peso o Peso_y
        materialRank = materialRank.sort_values(by=["Peso"],ascending=False)
        return materialRank
    
    def calculo_lista_top_libros(self,join, columna,baja_circulacion, media_circulacion, alta_circulacion):
        prestamos_recientes = join#.loc[(join.Year == 2020) | (join.Year == 2021)]
        #display(prestamos_recientes)
        frecuencia_prestamos = prestamos_recientes.groupby([columna])["Year"].count().reset_index(name="Frecuencia")
        #display(frecuencia_prestamos.Frecuencia.describe())
        primerPercentil = frecuencia_prestamos.Frecuencia.quantile(0.25)
        print("Percentil 1: ", primerPercentil)
        segundoPercentil = frecuencia_prestamos.Frecuencia.quantile(0.50)
        print("Percentil 2: ", segundoPercentil)
        tercerPercentil = frecuencia_prestamos.Frecuencia.quantile(0.75)
        print("Percentil 3: ", tercerPercentil)
        frecuencia_prestamos["Circulacion"] = 1
        frecuencia_prestamos.loc[(frecuencia_prestamos.Frecuencia <= segundoPercentil), "Circulacion"] = baja_circulacion
        frecuencia_prestamos.loc[(frecuencia_prestamos.Frecuencia > segundoPercentil) &
                                 (frecuencia_prestamos.Frecuencia <= tercerPercentil), "Circulacion"] = media_circulacion
        frecuencia_prestamos.loc[frecuencia_prestamos.Frecuencia > tercerPercentil, "Circulacion"] = alta_circulacion
        #display(frecuencia_prestamos)
        return frecuencia_prestamos
    
    def invocarCalculoLista(self):
        self.frecuencia_unidad = self.calculo_lista_top_libros(self.join,"DeweyUnidad", 10, 50,100)
        self.frecuencia_decena = self.calculo_lista_top_libros(self.join,"DeweyDecena", 50, 100, 150)
        self.frecuencia_centena = self.calculo_lista_top_libros(self.join,"DeweyCentena", 100, 150, 200)
        
        
    def generar_recomendacion(self,pesos_clustering, cluster, peso_x_nivel, total_recomendaciones, material_rankeado, columna,df_frecuencia):
        recomendaciones = pd.DataFrame(columns = ['IDUsuario', 'Llave', 'Nivel', 'Pertenencia'])
        #buscamos los prestamos asociados unicamente a un cluster
        prestamos_cluster = pesos_clustering.loc[pesos_clustering.Cluster == cluster]
        #Cambiamos el tipo de dato de pertenencia a float
        prestamos_cluster["Pertenencia"] = prestamos_cluster["Pertenencia"].astype(float) 
        #cada row es un usuario diferente
        for index, row in prestamos_cluster.iterrows():
            #calculamos el número de prestamos
            num_prestamos = math.ceil(row["Pertenencia"] * peso_x_nivel * total_recomendaciones)
            #obtenemos los prestamos del usuario de la tabla de join
            #buscamos que estos prestamos no se repitan
            prestamos_usuario = self.join.loc[(self.join["IDUsuario"] == row.IDUsuario) 
                                         & (self.join[columna] == row.Cluster) ]["Llave"].unique()
            tamanio_lista = df_frecuencia.loc[df_frecuencia[columna] == cluster]["Circulacion"].values[0]
            recomendaciones_usuario = material_rankeado[~material_rankeado.Llave.isin(prestamos_usuario)].head(tamanio_lista)
            try:
                llavesRecomendaciones = recomendaciones_usuario["Llave"].sample(n = num_prestamos)
            except:
                llavesRecomendaciones = recomendaciones_usuario["Llave"].head(num_prestamos)
            tamanio = len(llavesRecomendaciones)
            usuarios = np.repeat([row.IDUsuario], tamanio)
            nivel = np.repeat([peso_x_nivel], tamanio)
            pertenencia = np.repeat([row.Pertenencia], tamanio)
            aux_df = pd.DataFrame({'IDUsuario': usuarios,
                                   'Llave': llavesRecomendaciones.values,
                                   'Nivel': nivel,
                                   'Pertenencia': pertenencia})
            recomendaciones = pd.concat([recomendaciones, aux_df], ignore_index = True)

            ##Recomendacion libro de baja circulacion
            #obtenemos el dewey sobre el cual el usuario tiene mayor pertenencia
            mayor_dewey = self.eliminar_cero(row.IDUsuario, self.pesos_usuarios_unidad).drop('IDUsuario', axis=1).idxmax(axis=1).values[0]
            mayor_dewey = int(mayor_dewey)
            #Quitamos los libros que ya ha prestado el usuario
            if mayor_dewey == cluster and columna == "DeweyUnidad":
                material_rankeado2 = material_rankeado[~material_rankeado.Llave.isin(prestamos_usuario)]
                #Excluimos el top 50 libros de dicho dewey
                libros_baja_circulacion = material_rankeado2.iloc[tamanio_lista:]
                try:
                    llaveRecomendacion = libros_baja_circulacion["Llave"].sample(n =1).values[0]
                    aux_df = pd.DataFrame({'IDUsuario': row.IDUsuario, 'Llave': llaveRecomendacion, 'Nivel': "BC", 'Pertenencia': row.Pertenencia}, index=[0])
                    recomendaciones = pd.concat([recomendaciones, aux_df], ignore_index = True)
                except:
                    error = 1
                    #print("No hay libros de baja circulacion a recomendar para este usuario")
        return recomendaciones

    def recomendaciones_nivel(self,pesos_clustering, peso_x_nivel, total_recomendaciones, columna, df_frecuencia):#"DeweyUnidad"
        print("Comenzando Recomendaciones nivel ")
        recomendaciones = pd.DataFrame(columns = ['IDUsuario', 'Llave', 'Nivel'])
        i=0
        for cluster in pesos_clustering.Cluster.unique():
            #print("CLUSTER: ", cluster)
            material_rankeado = self.rank_material(pesos_clustering, cluster, "DeweyUnidad")
            #print(material_rankeado)
            lista_recomendaciones = self.generar_recomendacion(pesos_clustering,
                                                          cluster,
                                                          peso_x_nivel,
                                                          total_recomendaciones,
                                                          material_rankeado,
                                                          columna,
                                                        df_frecuencia)
            #print(lista_recomendaciones.shape[0])
            recomendaciones = pd.concat([recomendaciones, lista_recomendaciones], ignore_index = True)
            #display(recomendaciones)
            i=i+1
            if i%10 == 0:
                print(i)
        print("Finalizando Recomendaciones nivel")
        return recomendaciones
    
    def invocarRecomendaciones(self):
        self.recomendaciones_final_unidad = self.recomendaciones_nivel(self.pesos_clustering_unidad, 0.5, 10, "DeweyUnidad",self.frecuencia_unidad)
        self.recomendaciones_final_decena = self.recomendaciones_nivel(self.pesos_clustering_decena, 0.2, 10, "DeweyDecena",self.frecuencia_decena)
        self.recomendaciones_final_centena = self.recomendaciones_nivel(self.pesos_clustering_centena, 0.1, 10, "DeweyCentena",self.frecuencia_centena)

        
    def recomendar_nuevo(self,usuario, pesos_usuarios, libros_nuevos):
        #obtenemos el dewey sobre el cual el usuario tiene mayor pertenencia
        #sobre este dewey se recomendará el libro nuevo
        mayor_dewey = self.eliminar_cero(usuario, pesos_usuarios).drop('IDUsuario', axis=1).idxmax(axis=1).values[0]
        mayor_dewey = int(mayor_dewey)
        prestamos_usuario = self.join.loc[(self.join["IDUsuario"] == usuario) & (self.join["DeweyUnidad"] == mayor_dewey) ]["Llave"].unique()
        libros_nuevos_dewey = libros_nuevos.loc[libros_nuevos.DeweyUnidad == mayor_dewey]
        posibles_recomendaciones = libros_nuevos_dewey[~libros_nuevos_dewey.Llave.isin(prestamos_usuario)].head(50)
        try:
            llaveRecomendacion = posibles_recomendaciones["Llave"].sample(n =1).values[0]
        except:
            return -1
        return llaveRecomendacion

    def recomendaciones_libros_nuevos(self,pesos_usuarios):
        usuarios = self.join.IDUsuario.unique()
        recomendaciones = pd.DataFrame(columns = ['IDUsuario', 'Llave', 'Nivel', 'Pertenencia'])
        libros_nuevos = self.join.loc[(self.join.FechaCreacion > 2018)]
        for usuario in usuarios:
            llave = self.recomendar_nuevo(usuario, pesos_usuarios, libros_nuevos)
            if llave != -1:
                aux_df = pd.DataFrame({'IDUsuario': usuario, 'Llave': llave, 'Nivel': "Nuevo", 'Pertenencia': "Nuevo"}, index=[0])
                recomendaciones = pd.concat([recomendaciones, aux_df], ignore_index = True)
        return recomendaciones
    
    def invocarRecomendacionesLibrosNuevos(self):
        self.recomendaciones_nuevas = self.recomendaciones_libros_nuevos(self.pesos_usuarios_unidad)
    
    def recomendaciones_usuario(id_usuario, df_recomendaciones):
        recomendaciones_u = df_recomendaciones.loc[df_recomendaciones["IDUsuario"] == id_usuario]
        display(recomendaciones_u)
        return recomendaciones_u
    
    def unionRecomendaciones(self):
        self.recomendaciones_generales = pd.DataFrame(columns = ['IDUsuario', 'Llave', 'Nivel'])
        self.recomendaciones_generales = pd.concat([self.recomendaciones_generales, self.recomendaciones_final_unidad], ignore_index = True)
        self.recomendaciones_generales = pd.concat([self.recomendaciones_generales, self.recomendaciones_final_decena], ignore_index = True)
        self.recomendaciones_generales = pd.concat([self.recomendaciones_generales, self.recomendaciones_final_centena], ignore_index = True)
        self.recomendaciones_generales = pd.concat([self.recomendaciones_generales, self.recomendaciones_nuevas], ignore_index = True)

    
    def filtrar_recomendaciones(nivel,total_recomendaciones):
        num_rec = int(nivel*total_recomendaciones)
        recomendaciones_filtradas = pd.DataFrame()
        recomendaciones_nivel = self.recomendaciones_generales.loc[(recomendaciones_generales.Nivel == nivel)]    
        for usuario in self.recomendaciones_generales.IDUsuario.unique():
            rec = recomendaciones_nivel.loc[(recomendaciones_nivel.IDUsuario == usuario)]    
            if rec.shape[0] > num_rec:
                rec = rec.sample(n=num_rec)
            recomendaciones_filtradas = pd.concat([recomendaciones_filtradas,rec], ignore_index = True)
        return recomendaciones_filtradas
    
    
    def invocarFiltrarRecomendaciones(self):
        self.recomendaciones_finales_unidad = self.filtrar_recomendaciones(0.5,10)
        self.recomendaciones_finales_decena = self.filtrar_recomendaciones(0.2,10)
        self.recomendaciones_finales_centena = self.filtrar_recomendaciones(0.1,10)
        
    def unionRecomendacionesFinales(self):
        
        self.recomendaciones_bc = self.recomendaciones_generales.loc[self.recomendaciones_generales.Nivel == "BC"]
        self.recomendaciones_nuevas = self.recomendaciones_generales.loc[self.recomendaciones_generales.Nivel == "Nuevo"]
        
        self.recomendaciones_finales = pd.DataFrame()
        self.recomendaciones_finales = pd.concat([self.recomendaciones_finales, self.recomendaciones_finales_unidad], ignore_index = True)
        self.recomendaciones_finales = pd.concat([self.recomendaciones_finales, self.recomendaciones_finales_decena], ignore_index = True)
        self.recomendaciones_finales = pd.concat([self.recomendaciones_finales, self.recomendaciones_finales_centena], ignore_index = True)
        self.recomendaciones_finales = pd.concat([self.recomendaciones_finales, self.recomendaciones_nuevas], ignore_index = True)
        self.recomendaciones_finales = pd.concat([self.recomendaciones_finales, self.recomendaciones_bc], ignore_index = True)


    def recomendaciones_completas:
        info_libros = join[['Llave', 'DeweyUnidad', 'Titulo']].drop_duplicates(subset=['Llave'])
        recomendaciones_completas = recomendaciones_finales.merge(info_libros, on='Llave',how='inner')#, rsuffix='_right', lsuffix='_left')
        column_names = ["IDUsuario", "Llave", "Nivel", "Pertenencia"]
        recomendaciones_completas.to_json(r'C:\Users\user\Downloads\recomedaciones_completas.json')
   
        
    def exportarRecomendaciones(self):
        self.recomendaciones_finales.to_json(r'/Users/juansebastianangaritatorres/Downloads/recomedaciones_finales.json')
        
