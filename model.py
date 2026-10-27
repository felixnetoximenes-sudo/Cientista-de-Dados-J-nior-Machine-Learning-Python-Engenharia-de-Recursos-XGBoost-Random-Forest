import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
tabela = pd.read_csv('valores.csv')
tabela = tabela.drop_duplicates()


tabela['vote_count'] = pd.to_numeric(tabela['vote_count'], errors='coerce')
tabela['vote_average'] = pd.to_numeric(tabela['vote_average'], errors='coerce')
tabela['id'] = pd.to_numeric(tabela['id'], errors='coerce')
tabela['popularity'] = pd.to_numeric(tabela['popularity'], errors='coerce')


tabela = tabela.fillna(0)

x = tabela[['vote_count', 'vote_average', 'id']]
y = tabela['popularity']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, random_state=42)

modelo_xgb = XGBRegressor(max_depth=6, n_estimators=150, learning_rate=0.08, random_state=42)

modelo_xgb.fit(X_train, Y_train)
previsao = modelo_xgb.predict(X_test)
erro = mean_absolute_error(previsao, Y_test)

print(f'O erro da IA corrigida: {erro}')

joblib.dump(modelo_xgb, 'modelo_pronto.pkl')
print ('Modelo Salvo com sucesso')