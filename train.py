import joblib
import pandas as pd
from xgboost import XGBRegressor

tabela = joblib.load('modelo_pronto.pkl')


novo = pd.DataFrame([[100, 200, 311]], columns=['vote_count', 'vote_average', 'id'])


previsao = tabela.predict(novo)
print(f'Previsão da IA: {previsao}')
