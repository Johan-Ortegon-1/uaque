import pandas as pd
import re
import datetime



class Material():
    
    def __init__(self):
        pathMaterial =("https://www.dropbox.com/s/96kxr4dw10q7tzl/B-GENERAL-COLECCIONES-TEMATICAS.json?dl=1")
        self.datasetMaterial = pd.read_json(pathMaterial)


    def cleanDate(self):

        fechasFormat = self.datasetMaterial["Fecha de creación"].apply(lambda x: str(x))
        fechasSave = fechasFormat.apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d'))
        self.datasetMaterial["Fecha de creación"] = fechasSave


    def cleanKeys(self):
        self.datasetMaterial['Llaves'] = self.datasetMaterial['Llaves'].apply(lambda x: str(x))



    def cleanColumns(self):
        nombresNuevos = {'Id de item':'IDItem',
         'Ubicación Habitual':'UbicacionHabitual',
          'Llaves':'Llave',
         'Ubicación Actual':'UbicacionActual',
          'Fecha de creación':'FechaCreacion',
          'Numero de Ubicación':'NumeroUbicacion',
          'Total de préstamos':'TotalPrestamos',
          'Autor corporativo':'AutorCorporativo',
          'Tipo de ítem':'TipoItem',
          'Título':'Titulo',
         'Tema 650':'Tema650',
         'Todas las temáticas':'TodasTematicas',
         'Año publicación':'AnioPublicacion'}

        self.materialLimpio = self.datasetMaterial.rename(columns=nombresNuevos)
        self.materialLimpio = self.materialLimpio.drop(['UbicacionHabitual',
                                                'Biblioteca',
                                                'TodasTematicas',
                                                'AutorCorporativo',
                                                'Cat2',
                                                'Copias',
                                                'TotalPrestamos',
                                              'Vol'],axis=1)


    def cleanItemType(self):
        self.materialLimpio = self.materialLimpio[self.materialLimpio['TipoItem'].isin([
        'LIBRO',
        'LITERATURA',
        'REFERENCIA',
        'RESERVA',
        ])]


    def cleanSignature(self):
        self.materialLimpio["Signatura"] = self.materialLimpio["Signatura"].apply(lambda x: 
                               x.split()[0])


    def estandarizarDeweys(self, signatura): 
      match = re.search(r'\d{3}', signatura)
      if match:
        return match.group(0)
      else: 
        return signatura


    def cleanDeweys(self):
        self.materialLimpio["DeweyUnidad"] = self.materialLimpio["Signatura"].apply(lambda signatura: self.estandarizarDeweys(signatura))

        self.materialLimpio.loc[self.materialLimpio.DeweyUnidad == "44O.7", "DeweyUnidad"] = "440"
        self.materialLimpio.loc[self.materialLimpio.DeweyUnidad == "3O3.4", "DeweyUnidad"] = "303"
        self.materialLimpio.loc[self.materialLimpio.DeweyUnidad == "R", "DeweyUnidad"] = "-999"

        self.materialLimpio.drop(self.materialLimpio[self.materialLimpio.DeweyUnidad == 'L.V.'].index, inplace=True)
        self.materialLimpio.drop(self.materialLimpio[self.materialLimpio.DeweyUnidad == 'T.MEPG'].index, inplace=True)
        self.materialLimpio.drop(self.materialLimpio[self.materialLimpio.DeweyUnidad == 'F'].index, inplace=True)
        self.materialLimpio.drop(self.materialLimpio[self.materialLimpio.DeweyUnidad == 'CO'].index, inplace=True)
        self.materialLimpio.drop(self.materialLimpio[self.materialLimpio.DeweyUnidad == 'CUMANES'].index, inplace=True)
        self.materialLimpio.drop(self.materialLimpio[self.materialLimpio.DeweyUnidad == 'CD'].index, inplace=True)
        self.materialLimpio.drop(self.materialLimpio[self.materialLimpio.DeweyUnidad == 'M'].index, inplace=True)
        self.materialLimpio.drop(self.materialLimpio[self.materialLimpio.DeweyUnidad == 'E'].index, inplace=True)
        self.materialLimpio.drop(self.materialLimpio[self.materialLimpio.DeweyUnidad == 'INV'].index, inplace=True)


        self.materialLimpio['DeweyUnidad'] = self.materialLimpio['DeweyUnidad'].astype(int)

        self.materialLimpio["DeweyDecena"] = self.materialLimpio.DeweyUnidad.apply(lambda x: int(x/10)*10)

        self.materialLimpio['DeweyDecena'] = self.materialLimpio['DeweyDecena'].astype(int)

        self.materialLimpio["DeweyCentena"] = self.materialLimpio.DeweyDecena.apply(lambda x: int(x/100)*100)

        self.materialLimpio['DeweyCentena'] = self.materialLimpio['DeweyCentena'].astype(int)

    def cleanPublishYear(self):
        self.materialLimpio.loc[self.materialLimpio["AnioPublicacion"] >2021,"AnioPublicacion"] = 0
        self.materialLimpio.FechaCreacion.apply(lambda x: x.year)


    def clean(self):
        self.cleanDate()
        self.cleanKeys()
        self.cleanColumns()
        self.cleanItemType()
        self.cleanSignature()
        self.cleanDeweys()
        self.cleanPublishYear()





class History():
    
    def __init__(self):
        
        pathHistory =("https://www.dropbox.com/s/wpslbav7qanv4qt/PRESTAMOS-BGENERAL.json?dl=1")
        self.datasetPrestamos = pd.read_json(pathHistory)
        self.datasetPrestamos = self.datasetPrestamos[self.datasetPrestamos['ID de usuario ok'] != "69c8887e954b39caa50b2ad21c66bad0b1c7715a"]


    def cleanLibraries(self):
        self.datasetPrestamos = self.datasetPrestamos[self.datasetPrestamos.Biblioteca == 'B-GENERAL']
        self.datasetPrestamos = self.datasetPrestamos[self.datasetPrestamos["Biblioteca Transacción"] == 'B-GENERAL']
        
        
    def cleanDates(self):
        fechasFormat = self.datasetPrestamos['Fecha'].apply(lambda x: x[1:15])
        fechasSave = fechasFormat.apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d%H%M%S'))
        self.datasetPrestamos['Fecha'] = fechasSave
        self.datasetPrestamos['Year']=self.datasetPrestamos['Fecha'].apply(lambda x: str(x)[0:4])
        
        
    def cleanFaculty(self):     
        self.datasetPrestamos["Facultad"] = self.datasetPrestamos["Facultad"].str.lower()     
        
        
    def cleanColumns(self):          
        nombresNuevos = {'row ID':'RowID',
         'Transacción':'Transaccion',
         'Biblioteca Transacción':'BibliotecaTransaccion',
          'ID Ítem':'IDItem',
          'Numero de Ubicación':'NumeroUbicacion',
          'Ubicación':'Ubicacion',
          'Tipo de ítem':'TipoItem',
          'Llaves':'Llave',
         'Categoría 1':'Caterogoria1',
         'ID de usuario ok':'IDUsuario'}

        self.prestamosLimpio = self.datasetPrestamos.rename(columns=nombresNuevos)

        self.prestamosLimpio = self.prestamosLimpio.drop(['BibliotecaTransaccion',
                                                'Biblioteca',
                                                'Hora',
                                                'Transaccion',
                                                'Caterogoria1'],axis=1)        
        
        
        
    def cleanItemType(self):
        self.prestamosLimpio = self.prestamosLimpio[self.prestamosLimpio['TipoItem'].isin([
        'LIBRO',
        'LITERATURA',
        'REFERENCIA',
        'RESERVA',
        ])]
        
    def clean(self):
        self.cleanLibraries()
        self.cleanDates()
        self.cleanColumns()
        self.cleanColumns()
        self.cleanItemType()





class Merger():

    
    def __init__(self, df_prestamos, df_material):
        self.datasetPrestamos = df_prestamos
        self.datasetMaterial = df_material
        
        self.datasetPrestamos['IDItem'] = self.datasetPrestamos['IDItem'].apply(lambda x: str(x))
        self.datasetMaterial['IDItem'] = self.datasetMaterial['IDItem'].apply(lambda x: str(x))



    def join(self):
        dpf_data = pd.DataFrame(data=self.datasetPrestamos)
        dmf_d_data = pd.DataFrame(data=self.datasetMaterial)
        join_dm_dp = pd.merge(dpf_data, dmf_d_data, left_on='IDItem', right_on='IDItem', how='left')

        eliminar = join_dm_dp.loc[join_dm_dp["Titulo"].isnull()]
        self.df = join_dm_dp[~join_dm_dp.index.isin(eliminar.index)]
        
        self.df = self.df.drop(['Llave_y','TipoItem_x','TipoItem_y','UbicacionActual','Perfil','Tema650','Mes', 'NumeroUbicacion', 'Ubicacion', 'FechaCreacion', 'Dewey'],axis = 1)
        self.df = self.df.rename(columns={'Dewey_x':'Dewey', 'Llave_x':'Llave','Dewey_y':'DeweyEspecifico'})
    
        # Feedback
        basepath = "https://www.dropbox.com/s/"

        # Dataframe que contiene la información del feedback de los usuarios
        lib_feedback = pd.read_json(basepath + "gk705nycjc5hza0/feedback_users.json?dl=1")
        
        self.df = self.df.append(lib_feedback, ignore_index=True)






















