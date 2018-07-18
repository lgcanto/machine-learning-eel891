#Importação de pacotes necessários para o script
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
#import xgboost as xgb
#import lightgbm as lgb

warnings.filterwarnings('ignore')

#Carrega o set de treino
df_train = pd.read_csv('kaggle_data/train.csv')
#Carrega o set de teste
df_test = pd.read_csv('kaggle_data/test.csv')

print("\n")
#Informações estatísticas dos preços
print("Informações estatísticas dos preços:\n")
print(df_train['SalePrice'].describe())

test_ID = df_test['Id']


### ANÁLISE DE DADOS ###


print("\n")
#Matriz de correlação com as dez variáveis com maior índice de correlação
corrmat = df_train.corr()
print("Matriz de correlação com as dez variáveis com maior índice de correlação:\n")
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


### LIDANDO COM VARÍAVEIS SEM DADOS ###


print("\n")
#Variáveis sem dados (nulas)
print("Variáveis sem dados:\n")
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))


#Lidando com os dados perdidos (deleta alguns dados de df_train!)

#df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
#df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
#df_train.isnull().sum().max() #Checando se não existem mais dados nulos


### REMOÇÃO DE OUTLIERS ###


print("\n")
print("Remoção de outliers:\n")
print("Antes da remoção:\n")

#Análise SalePrice/GrLivArea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#deletando outliers
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

print("Depois da remoção:\n")
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#Análise SalePrice/TotalBsmtSF
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


### NORMALIZAÇÃO ###


#Plot do histograma e da normal de SalePrice
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)

#Aplicando transformação logarítma
df_train['SalePrice'] = np.log(df_train['SalePrice'])

#Replotando
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


### LIDANDO COM VARIÁVEIS DE CATEGORIA ###

#Função de validação
n_folds = 5

x_train = df_train
x_test = df_test

y_train = df_train.SalePrice.values
x_train.drop(['SalePrice'], axis=1, inplace=True)
x_train.drop("Id", axis = 1, inplace = True)

x_test.drop("Id", axis = 1, inplace = True)

x_train = x_train.fillna('None')
x_test = x_test.fillna('None')

#Converte as variáveis para variáveis dummy (0 ou 1 para cada tipo de categoria)
x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train.values)
    rmse= np.sqrt(-cross_val_score(model, x_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
    
lasso = make_pipeline(StandardScaler(), Lasso(alpha =0.0005, random_state=1))

#Score do modelo com o dataset de treino
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


#Treinamento e predisão  final

#total = df_test.isnull().sum().sort_values(ascending=False)
#percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)
#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#print(missing_data.head(25))
#df_test = df_test.drop((missing_data[missing_data['Total'] > 0]).index,1)
#df_test.isnull().sum().max() 


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

lasso.fit(x_train.values, y_train)
train_pred = lasso.predict(x_train.values)
test_pred = lasso.predict(x_test.values)
print(rmsle(y_train, train_pred))