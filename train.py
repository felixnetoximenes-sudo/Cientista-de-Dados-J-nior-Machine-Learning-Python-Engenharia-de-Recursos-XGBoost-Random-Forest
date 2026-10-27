import joblib
import pandas as pd
from xgboost import XGBRegressor

tabela = joblib.load('modelo_pronto.pkl')

# 2. Criamos o dado novo passando os nomes exatos das colunas do treino
novo = pd.DataFrame([[100, 200, 311]], columns=['vote_count', 'vote_average', 'id'])

# 3. Agora o XGBoost aceita e faz a mágica acontecer!
previsao = tabela.predict(novo)
print(f'Previsão da IA: {previsao}')
