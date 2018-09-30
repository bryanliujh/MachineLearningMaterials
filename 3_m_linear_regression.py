import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

dataset = pd.read_csv('datafiles/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding dummy variable trap (unecessary lib will take care)
X = X[:, 1:]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)


regressor = LinearRegression()
regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)

#compare y_test with y_predict
print(y_pred)
print(y_test)


#bulding the optimal model with backward elimination
#need to add a column of 1 (X0 constant) as sm lib does not include intercept
X = np.append(arr = np.ones((50,1)).astype(int), values=X,axis=1)
# X_opt will only contain optimal variable with high impact on X
#fit the full model with all possible predictor
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())

#since pvalue of x2 is above sig of 0.05, remove x2
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())

#since pvalue of x1 is above sig of 0.05, remove x1
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())

#since pvalue of x2 is above sig of 0.05, remove x2 (which is 4)
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())

#since pvalue of x2 is above sig of 0.05, remove x2 (which is 5)
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())

#conclusion R&D spent is the best variable


#automatic backward elimination with p values
'''

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x


SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
'''
#automatic backward elimination with p values & adjusted r squared

'''
import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
'''