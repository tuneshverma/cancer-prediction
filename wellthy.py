
# coding: utf-8

# ### load libraries

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns


# ### load data
data_1 = pd.read_csv('D:/datasets/wellthy/CAX_CANCER_Updated_TRAIN_data.csv')

corr_matrix=data_1.corr() #to check correlation between "TD50_Rat_mg" and other features
k=corr_matrix['TD50_Rat_mg'].sort_values(ascending=False)

# ###### Some features gives the result as "NaN" that means they are not correlated at all so remove it

data = data_1.drop(['SssNH2count','SsssNHcount','SddsN(nitro)count','SsPH2count','SssPHcount','SsssPcount',
                  'SsssssPcount','SssNH2E-index','SsssNHE-index','SddsN(nitro)E-index','SsPH2E-index',
                  'SssPHE-index','SsssPE-index','SsssssPE-index','T_2_F_0','T_C_F_0','T_C_S_0','T_C_Cl_0',
                  'T_N_F_0','T_N_F_1','T_N_F_2','T_N_S_0','T_N_Cl_0','T_N_Cl_1','T_O_F_0','T_O_F_1','T_O_S_0',
                  'T_O_Cl_0','T_F_F_1','T_F_F_7','T_F_S_0','T_F_S_1','T_F_S_3','T_F_S_4','T_F_S_6','T_F_S_7',
                  'T_F_Cl_0','T_F_Cl_1','T_F_Cl_4','T_F_Cl_6','T_F_Cl_7','T_S_Cl_0','T_S_Cl_1','T_Cl_Cl_1'], axis=1)   


corr_matrix=data.corr() #to check correlation between "TD50_Rat_mg" and other features
l=corr_matrix['TD50_Rat_mg'].sort_values(ascending=False)

data.describe()

# ## Data visualization

plt.scatter(data['TD50_Rat_mg'], data['Volume'], s=None, c=None, marker=None, cmap=None, norm=None, vmin=None,
            vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, data=None)
plt.show()


# ##### from above graph we get the insight that those two features are not too correlated 

data.plot.scatter(x='TD50_Rat_mg', y='Mol.Wt.')
plt.show()

# ##### from above graph we get the insight that those two features are not too correlated 

plt.scatter(data['Volume'], data['Mol.Wt.'])
plt.show()


# ##### while on the other hand "volume" and "Mol.Wt." are highly correlated

sns.distplot(data['TD50_Rat_mg'], bins=10, kde=False)
plt.show()


# ###### the label data is non-uniform maximum values are near zero

a=data.groupby(['TD50_Rat_mg']).mean()[['Mol.Wt.','Volume']]
a.plot.line()
plt.show()

b=data.groupby(['TD50_Rat_mg']).mean()[['H-AcceptorCount','H-DonorCount','RotatableBondCount']]
b.plot.line()
plt.show()

c=data.groupby(['TD50_Rat_mg']).mean()[['chi0','chi1','chi2','chi3','chi4','chi5',
                                        'chiV0','chiV1','chiV2','chiV3','chiV4','chiV5']]
c.plot.line()
plt.show()

d=data.groupby(['TD50_Rat_mg']).mean()[['kappa1','kappa2','kappa3','k1alpha','k2alpha','k3alpha']]
d.plot.line()
plt.show()

e=data.groupby(['TD50_Rat_mg']).mean()[['HydrogensCount','CarbonsCount','SulfursCount','OxygensCount',
                                        'NitrogensCount','ChlorinesCount','FluorinesCount','BrominesCount']]
e.plot.line()
plt.show()


# ##### in above line plots it looks like those features are highly correlated and its similar high peaks when the value of "TD50" around 5000


f=data.groupby(['TD50_Rat_mg']).mean()[['T_O_S_5','SddssS(sulfate)count','T_2_O_6','T_O_S_1','T_2_O_5','T_O_O_6',
                                        'Quadrupole3','SsCH3count','YcompDipole',
                                        'XcompDipole','SddssS(sulfate)E-index']]
f.plot.line()
plt.show()


# ##### line plot between the most correlated features, values of features are high around low value of "TD50" and high again around 25000 value of 'TD50

sns.distplot(data['T_O_S_5'], bins=10, kde=False)
plt.show()


# ##### count graph for the most correlated feature "T_O_S_5", maximum values is around Zero

sns.distplot(data['SddssS(sulfate)E-index'], bins=10, kde=False)
plt.show()

# ##### count graph for the other most correlated feature "SddssS(sulfate)E-index", maximum values is around Zero

test_1 = pd.read_csv('D:/datasets/wellthy/CAX_CANCER_TEST_data.csv')


# In[190]:


test = test_1.drop(['SssNH2count','SsssNHcount','SddsN(nitro)count','SsPH2count','SssPHcount','SsssPcount',
                  'SsssssPcount','SssNH2E-index','SsssNHE-index','SddsN(nitro)E-index','SsPH2E-index',
                  'SssPHE-index','SsssPE-index','SsssssPE-index','T_2_F_0','T_C_F_0','T_C_S_0','T_C_Cl_0',
                  'T_N_F_0','T_N_F_1','T_N_F_2','T_N_S_0','T_N_Cl_0','T_N_Cl_1','T_O_F_0','T_O_F_1','T_O_S_0',
                  'T_O_Cl_0','T_F_F_1','T_F_F_7','T_F_S_0','T_F_S_1','T_F_S_3','T_F_S_4','T_F_S_6','T_F_S_7',
                  'T_F_Cl_0','T_F_Cl_1','T_F_Cl_4','T_F_Cl_6','T_F_Cl_7','T_S_Cl_0','T_S_Cl_1','T_Cl_Cl_1'], axis=1)

data = data.fillna(data.mean(), inplace = True)

test = test.fillna(test.mean(), inplace = True)

data
#test

x_data = data.drop(['TD50_Rat_mg'], axis=1) #Features

y_data = data['TD50_Rat_mg'] #labels

# split the data in training and validation set
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.40, random_state=42) 

#x_val

x_test = test.drop(['TD50'], axis=1)

#x_test

# scaling the data to feed it to algo
x_train_scale = preprocessing.scale(x_train)
x_val_scale = preprocessing.scale(x_val)



x_train_scale=pd.DataFrame(x_train_scale)
x_val_scale = pd.DataFrame(x_val_scale)


y_train_scale = preprocessing.scale(y_train)
y_val_scale = preprocessing.scale(y_val)

y_train_scale=pd.DataFrame(y_train_scale)
y_val_scale=pd.DataFrame(y_val_scale)


# ##### This data has 845 rows × 485 columns. So many features causes curse of dimentionallity so to decrease the number of features we will use PCA

pca = PCA(n_components=338)
x_train_tran=pca.fit_transform(x_train_scale)
x_val_tran=pca.fit_transform(x_val_scale)


x_train_tran = pd.DataFrame(x_train_tran)
x_val_tran = pd.DataFrame(x_val_tran)

# #### Random Forest

rf_regr = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=1500, n_jobs=-1)
rf_regr.fit(x_train_tran, y_train_scale)
rf_pred = rf_regr.predict(x_val_tran)
r2_score(y_val_scale, rf_pred)


# #### SVM

svm_regr = SVR(kernel='rbf', C=10, gamma=10, degree=5)
svm_regr.fit(x_train_tran, y_train_scale)
svm_pred = svm_regr.predict(x_val_tran)
r2_score(y_val_scale, svm_pred)


# #### Adaboost 

ada_regr = AdaBoostRegressor(base_estimator=None, n_estimators=100, learning_rate=0.01, loss='linear', random_state=None)
ada_regr.fit(x_train_tran, y_train_scale)
ada_pred = ada_regr.predict(x_val_tran)

r2_score(y_val_scale, ada_pred)


# ### it looks like Random Forest is giving the best results

# ### There are many features that are predictive in nature in this dataset, features values that has more then 75% of data zero are hard to predict
