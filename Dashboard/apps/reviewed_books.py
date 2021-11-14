import pandas as pd
from .feedbacks import feedbacks
from .joinTablas import joinTablas

#Traemos todas las llaves con susu deweys de todas las unidades
all_deweys = pd.DataFrame(joinTablas[['DeweyUnidad', 'DeweyDecena', 'DeweyCentena', 'Llave']]
.drop_duplicates())

#Join entre las dos tablas desde la Llave del libro
reviewed_books: pd.DataFrame = feedbacks.merge(all_deweys, on='Llave', suffixes=('_feedback', '_all_deweys'))
reviewed_books = pd.DataFrame(reviewed_books.drop_duplicates(subset=['IDUsuario', 'Calificacion', 'Llave']))

