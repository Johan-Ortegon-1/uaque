import pandas as pd
print('Importando recomendaciones_completas')
recomendaciones_completas: pd.DataFrame = pd.DataFrame(pd.read_json('apps/recomedaciones_completas.json'))
print('Completado')
