# 🎬 Pipeline Preditivo de Popularidade de Mídias com XGBoost

Este repositório contém um projeto completo de Machine Learning de ponta a ponta focado na previsão do índice de popularidade de títulos de mídia com base em métricas de engajamento do público (como volume de votos e avaliação média). 

O grande diferencial deste projeto foi a aplicação prática de técnicas cíclicas de otimização, evoluindo de uma abordagem inicial (Linear Regression) para modelos avançados de ensemble (**Random Forest** e **XGBoost**), culminando na redução expressiva da taxa de erro do modelo.

## 🚀 Funcionalidades e Etapas do Pipeline

* **Tratamento e Higienização de Dados:** Utilização da biblioteca `Pandas` para eliminação de registros duplicados e imputação de valores ausentes (`fillna`).
* **Validação Estratégica:** Divisão controlada da base de dados em conjuntos de treino e teste (proporção 80/20) com fixação de semente aleatória (`random_state=42`) para garantir a reprodutibilidade dos testes.
* **Benchmarking de Modelos:** Experimentação prática comparando a performance de algoritmos lineares com os motores `Random Forest Regressor` e `XGBoost Regressor`.
* **Otimização Avançada (Abordagem Nível Pleno):** Implementação de `GridSearchCV` combinada com **Validação Cruzada (Cross-Validation)** em 3 dobras para automatizar a busca pelos hiperparâmetros ótimos (profundidade de árvore e número de estimadores), mitigando riscos de *overfitting*.
* **Serialização para Produção (Deploy):** Salvamento do modelo campeão em um arquivo físico utilizando a biblioteca `Joblib`, permitindo que o cérebro da IA seja consumido por aplicações externas em tempo real sem a necessidade de re-treinamento.

## 📈 Evolução dos Resultados e Métricas (Métricas de Erro)

O desempenho do modelo foi avaliado utilizando a métrica **MAE (Erro Médio Absoluto)**. O processo iterativo de testes demonstrou a importância do refinamento dos dados e parâmetros:

* **Tentativa 1 (Baseline - Regressão Linear):** Erro MAE de `19.22`
* **Tentativa 2 (Ajuste de Algoritmo):** Erro MAE reduzido para `17.74`
* **Tentativa 3 (Ajuste Fino de Hiperparâmetros):** **Recorde alcançado com MAE de `15.82`** (Utilizando Random Forest controlado com `max_depth=5`).
* **Tentativa 4 (Validação com XGBoost):** Estabilização do erro em `16.17`, validando a consistência preditiva das variáveis escolhidas.

## 🛠️ Tecnologias e Bibliotecas Utilizadas

* **Python** (Linguagem base)
* **Pandas** (Manipulação e tratamento de dados tabulares)
* **NumPy** (Operações matemáticas e arrays)
* **Scikit-Learn** (Métricas de erro, divisão de dados e busca de hiperparâmetros)
* **XGBoost** (Algoritmo de Gradient Boosting de alta performance)
* **Joblib** (Persistência e salvamento de modelos em disco)

## 💻 Como Executar e Consumir o Modelo Salvo

### 1. Para treinar e otimizar o modelo:
Certifique-se de ter o arquivo `valores.csv` no mesmo diretório e execute o script de treinamento para gerar o arquivo `melhor_modelo_pleno.pkl`.

### 2. Para consumir a IA em produção (Simulação de sistema real):
```python
import joblib
import pandas as pd

# Carrega o cérebro da IA previamente salvo
modelo = joblib.load('melhor_modelo_pleno.pkl')

# Cria os dados de um novo filme recebido pelo sistema (vote_count, vote_average, id)
novo_filme = pd.DataFrame([[100, 200, 311]], columns=['vote_count', 'vote_average', 'id'])

# Executa a previsão instantânea
previsao = modelo.predict(novo_filme)
print(f'Previsão de Popularidade da IA: {previsao}')
```

---
*Projeto desenvolvido com o objetivo de aplicar metodologias ágeis de Machine Learning para a resolução de problemas reais de negócios com dados tabulares.*
