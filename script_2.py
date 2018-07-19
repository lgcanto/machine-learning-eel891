import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import norm, skew
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #Ignora warnings do sklearn e do seaborn
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limitando floats para 3 casa decimais

#Carregando datasets de treino e teste
train = pd.read_csv('kaggle_data/train.csv')
test = pd.read_csv('kaggle_data/test.csv')

#Salvando a coluna de Id, pois esta será retirada
train_ID = train['Id']
test_ID = test['Id']

#Retirando coluna de Id dos datasets
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#Verificando a matriz de correlação com as 10 variáveis mais correlatas à SalePrice
corrmat = train.corr()
numVariaveis = 10
cols = corrmat.nlargest(numVariaveis, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#Remoção de Outliers
#Verificamos que OverallQual é a variável com maior correlação. Mas como esta é 
#uma variável categórica, a remoção de outliers irá acontecer apenas na segunda
#variável mais correlata, isto é, GrLivArea

#Verificando os outliers de GrLivArea:
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

#Removendo os outliers, isto é, os pontos com SalePrice abaixo de 300000 e
#GrLivArea acima de 4000:
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Verificando a ausência dos OutLiers:
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


#Normalização de SalePrice
#Histograma e probabilidade:
sns.distplot(train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)

#Aplicando transformação logarítma:
train['SalePrice'] = np.log(train['SalePrice'])

#Replotando:
sns.distplot(train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)

#Manipulação de features:
#Como a manipulação de varíaveis categóricas e dados nulos deve ser feito
#tanto no dataset de treino quanto no de test, os datasets serão concatenados
#para uma maior economia de código:
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
#Removendo coluna SalePrice que não está presente no dataset de teste:
all_data.drop(['SalePrice'], axis=1, inplace=True)

#Dados nulos:
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Porcentagem de nulos' :all_data_na})
missing_data.head(20)



#PULANDO ALGUMAS ETAPAS AFIM DE TESTES#
all_data = all_data.fillna("None")
all_data = pd.get_dummies(all_data)
train = all_data[:ntrain]
test = all_data[ntrain:]
#PULANDO ALGUMAS ETAPAS AFIM DE TESTES#



#Função de validação RMSE
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

lasso.fit(train.values, y_train)
lasso_train_pred = lasso.predict(train.values)
lasso_pred = np.expm1(lasso.predict(test.values))
print(rmsle(y_train, lasso_train_pred))

sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = lasso_pred
sub.to_csv('submission.csv',index=False)