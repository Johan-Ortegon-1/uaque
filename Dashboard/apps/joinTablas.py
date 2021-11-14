import pandas as pd
print('Importando joinTablas')
joinTablas: pd.DataFrame = pd.DataFrame(pd.read_json('apps/joinTablas.json'))
print('Completado')
