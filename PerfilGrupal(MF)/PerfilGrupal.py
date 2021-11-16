import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
import datetime
import gzip, pickle, pickletools
import time

class PerfilGrupal:
    
    def __init__(self):
        print("inicializando")
        self.join = self.cargaDatos()
    
    def cargaDatos(self):
        #importamos la tabla de join 
        join = pd.read_json('https://www.dropbox.com/s/i1komhf7u1c4y95/joinTablas.json?dl=1')
        #Eliminamos algunas columnas que no nos interesan para este notebook
        join = join.drop(["Fecha","Dewey","Facultad","Temas","Union","TipoItem"], axis=1)
        return join[:1000]
        
        
    def crearPesos(self):
        now = datetime.datetime.now()
        anio_actual = int(now.year)
        #Creamos la columna pesos para el dataframe a partir del año en que se realizó el prestamo
        self.join["Peso"] = self.join.apply(lambda row: 1/2**(anio_actual-row.Year), axis=1 )
        self.join[["Year","Peso"]]
        self.join.loc[self.join.calificacion.isnull(), "calificacion"] = 1.0
        self.join["Peso"] = self.join.apply(lambda row: (row.calificacion*row.Peso), axis=1 )
        
        
    #crearTablaPesos: crea la tabla de pesos por usuario
    #parámetros
        #columna = {deweyUnidad|deweyDecena|deweyCentena}
    def crearTablaPesos(self,columna):
        #Tabla = pesos_usuarios
            #columnas = deweys
            #filas = usuarios
            #dato = peso que tiene el usuario en dicho dewey
        #creamos las columnas a partir de los deweys diferentes. 
        agrupacion = self.join.groupby(["IDUsuario",columna])["Peso"].sum().reset_index(name="Peso")
        display(agrupacion.head(5))
        
        #Los pesos no deben ser menores a cero
        agrupacion.loc[agrupacion.Peso < 0.0, "Peso"] = 0
        
        #cración del dataframe
        pesos_usuarios = pd.DataFrame()
        #Dataframe auxiliar para acelerar el proceso de concat
        aux = pd.DataFrame()

        ids = agrupacion["IDUsuario"].unique()
        print("Total de IDs de usuarios: ", len(ids))
        #Recorremos cada uno de los usuarios
        for usuario in ids:
            #obtenemos todos los prestamos del usuario
            prestamos = agrupacion.loc[agrupacion["IDUsuario"]==usuario]
            columnas = prestamos[columna].values
            pesos = prestamos["Peso"].values
            fila = pd.DataFrame(data = [pesos], columns = columnas)
            fila["IDUsuario"] = usuario
            aux = pd.concat([aux,fila])
            #cada 100 registros concatenamos al dataframe general
            if(aux.shape[0] == 100):
                pesos_usuarios = pd.concat([pesos_usuarios,aux])
                aux = pd.DataFrame()
        pesos_usuarios = pd.concat([pesos_usuarios,aux])
        #Cambiamos los nil por 0
        pesos_usuarios = pesos_usuarios.fillna(0)
        pesos_usuarios.reset_index(drop=True, inplace=True)
        return pesos_usuarios

    def invocarTablaPesos(self):
        self.pesos_usuarios_unidad= self.crearTablaPesos("DeweyUnidad")
        self.pesos_usuarios_decena = self.crearTablaPesos("DeweyDecena")
        self.pesos_usuarios_centena = self.crearTablaPesos("DeweyCentena")

        
        #función: normalizar las filas por valores entre cero y uno.
        #Parámetros: pesos_usuario, representa la matriz dispersa que se va a normalizar.
        #{pesos_usuarios_unidad|pesos_usuarios_decena|pesos_usuarios_centena}
    def normalizar_pesos(self,pesos_usuario):
        print("Normalizando pesos...Iniciando")
        usuarios = pesos_usuario['IDUsuario']
        pesos_usuario = pesos_usuario.apply(pd.to_numeric, errors='coerce').drop(["IDUsuario"],axis=1)
        #display(pesos_usuario)
        #Normalizamos por fila
        sumatoria = pesos_usuario.max(axis=1)
        pesos_norm = pesos_usuario.div(pesos_usuario.sum(axis=1), axis=0)
        pesos_norm_id = pesos_norm.copy()
        # a la tabla le agregamos la columna de IDUsuario para poder 
        #identificar que pesos son de cada usuario
        if len(pesos_norm_id) == len(self.join.IDUsuario.unique()):
            pesos_norm_id['IDUsuario'] = usuarios
        print("Normalizando pesos...Acabado")
        return pesos_norm_id
    
    def invocarNormalizarPesos(self):
        self.pesos_norm_id_unidad = self.normalizar_pesos(self.pesos_usuarios_unidad)
        self.pesos_norm_id_decena = self.normalizar_pesos(self.pesos_usuarios_decena)
        self.pesos_norm_id_centena = self.normalizar_pesos(self.pesos_usuarios_centena)

    #Elimina los valores diferentes de cero para una fila especifica
    #Sirve para visualizar únicamente los deweys donde el usuario tiene prestamos. 
    def eliminar_cero(self,id_usuario, df_pesos):
        m1 = (df_pesos['IDUsuario'] == id_usuario)
        m2 = (df_pesos[m1] != 0).all()
        return df_pesos.loc[m1,m2]
    
    #Esta función asocia una grado de pertenencia
    #Parámetros:
        #pesos_usuarios: matrix dispersa de pesos 
        #{pesos_usuarios_unidad|pesos_usuarios_decena|pesos_usuarios_centena}
        #nombre_archivo: nombre del archivo a exportar con la tabla de df_pertenencia
    #se crea la tabla df_pertenencia: describe la pertenecia de cada usuario a un cluster(dewey)
    def clustering(self,pesos_usuarios):
        print("Comenzando clustering...")
        #normalizamos los pesos
        pesos_norm_id_unidad = self.normalizar_pesos(pesos_usuarios)
        #ponemos el usuario a ambos dataframes
        pesos_usuarios["IDUsuario"] = pesos_norm_id_unidad["IDUsuario"]
        #Creamos la estructura del dataframe
        df_pertenencia = pd.DataFrame(columns = ['IDUsuario', 'Cluster', 'Pertenencia','Peso'])
        #iteramos por cada fila(usuario) del dataframe
        for index, row in pesos_norm_id_unidad.iterrows():
            #Extraemos los usuarios
            usuario = row["IDUsuario"]
            #Extraemos los pesos deiferentes de cero normalizados y sin normalizar
            usuario_limpio_norm = self.eliminar_cero(row["IDUsuario"], pesos_norm_id_unidad)
            usuario_limpio_orig = self.eliminar_cero(row["IDUsuario"], pesos_usuarios)
            pertenecias = usuario_limpio_norm.drop(["IDUsuario"], axis=1).values
            pesos = usuario_limpio_orig.drop(["IDUsuario"], axis=1).values
            #los clusters son las columnas de estos dataframes
            clusters = usuario_limpio_norm.drop(["IDUsuario"], axis=1).columns
            #creamos un array que tenga las veces necesarias al usuario
            tamanio = len(clusters)
            usuarios = np.repeat([usuario], tamanio)
            #agregamos la fila al dataframe
            fila = pd.DataFrame(data = {'IDUsuario': usuarios, 'Cluster': clusters
                                       , 'Pertenencia': pertenecias[0], 'Peso': pesos[0]})
            df_pertenencia = df_pertenencia.append(fila)
        print("Finalizando clustering...")
        return df_pertenencia
    
    def invocarClustering(self):
        self.pesos_clustering_unidad = self.clustering(self.pesos_usuarios_unidad)
        self.pesos_clustering_decena = self.clustering(self.pesos_usuarios_decena)
        self.pesos_clustering_centena = self.clustering(self.pesos_usuarios_centena)
        self.pesos_clustering_unidad.reset_index(drop=True, inplace=True)
        self.pesos_clustering_decena.reset_index(drop=True, inplace=True)
        self.pesos_clustering_centena.reset_index(drop=True, inplace=True)
        
    def exportarDatos(self):
        self.pesos_clustering_unidad.to_json(r'C:\Users\user\Downloads\pesos_clustering_unidad.json')
        self.pesos_clustering_decena.to_json(r'C:\Users\user\Downloads\pesos_clustering_decena.json')
        self.pesos_clustering_centena.to_json(r'C:\Users\user\Downloads\pesos_clustering_centena.json')
        
    def exportarModelo(self):
        now_time = time.strftime("%m%d%H%m")
        filepath = r"C:\Users\user\Downloads\Tree_trained_model_"+now_time+".pkl"
        with gzip.open(filepath, "wb") as f:
            pickled = pickle.dumps(self, protocol=4)
            optimized_pickle = pickletools.optimize(pickled)
            f.write(optimized_pickle)
        
        
        