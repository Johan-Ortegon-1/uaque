import pandas as pd
#Traer pesos de los usuarios (suman 1)
print('Importando pesos usuarios')
pesos_usuario: pd.DataFrame = pd.DataFrame(pd.read_json('apps/pesos_norm_id_unidad2.json'))
print('Completado')

