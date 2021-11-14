
import pandas as pd
print('Importando feedbacks')
feedbacks: pd.DataFrame = pd.DataFrame(pd.read_json('apps/feedback_users.json'))
print('Completado')

