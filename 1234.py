# Logistic Regression

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.model_selection as skm
# Importing the datasets

datasets = pd.read_csv('/Users/karthik/Desktop/MS-M/SEM2/OMS/HomeWork/HW3/Social_Network_Ads.csv')
print(datasets.head())
X = datasets.iloc[ : , [2,3]].values
print(X)
Y = datasets.iloc[:, 4].values
print(Y)

# Splitting the dataset into the Training set and Test set

#from sklearn.model_selection import train_test_split
print(len(datasets))
X_Train, X_Test, Y_Train, Y_Test = skm.train_test_split(X, Y, test_size = 0.25)
print(len(X_Train))
print(len(X_Test))
# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Fitting the Logistic Regression into the Training set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_Train, Y_Train)
#from sklearn.metrics import classification_report
#print(classification_report(Y_Test, Y_Pred))

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

import statsmodels.api as sm
logit_model=sm.Logit(Y_Train,X_Train)
result=logit_model.fit()
print(result.summary())

# Predicting the test set results

Y_Pred = classifier.predict(X_Test)
print(Y_Pred)
# Making the Confusion Matrix 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test, Y_Pred)

# Visualising the Training set results 

from matplotlib.colors import ListedColormap
X_Set, Y_Set = X_Train, Y_Train
X1, X2 = np.meshgrid(np.arange(start = X_Set[:,0].min() -1, stop = X_Set[:, 0].max() +1, step = 0.01),
                     np.arange(start = X_Set[:,1].min() -1, stop = X_Set[:, 1].max() +1, step = 0.01))

plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X2.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j,1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression ( Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#Exploratory Data Analysis
#Create a histogram of the Age
sns.distplot(datasets['Age'], bins=20)
#Create a jointplot showing Area Income versus Age.
sns.jointplot(datasets['EstimatedSalary'], datasets['Age'], kind= 'kde')
#Create a pairplot with the hue defined by the 'Clicked on Ad' column feature.
sns.pairplot(datasets, hue='Purchased')

