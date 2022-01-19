import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def ejemplo_linea():
    x= np.array([1,2,3,4], dtype='float')
    y= x
    w= -5
    m= 4
    a= 0.5
    ls= 1e10
    i=0

    while ls > 0.00001:
        hx = w*x
        ls = 1/(2*m) * ( (hx - y)**2 ).sum()
        sl = 1/(2*m) * ( (hx - y)*x ).sum()
        w = w - a * sl
        i += 1

    print(i)
    print(w)

def ejemplo_vino():
    dataset = pd.read_csv('C:/Users/Droidex/Dropbox/Documentos/Diplomado/Parte 4/db/wine.csv')

    #eliminar nulos
    dataset.isnull().any()
    dataset = dataset.fillna(method='ffill')

    #llenar los datos de las columnas
    cnames = ['fixed acidity', 'volatile acidity', 'citric acid',
              'residual sugar', 'chlorides', 'free sulfur dioxide',
              'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    
    X = dataset[cnames].values
    y = dataset['quality'].values

    #plt.figure(figsize=(15,10))
    #plt.tight_layout()
    #seabornInstance.displot(dataset['quality'])

    #plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    coeff_df = pd.DataFrame(regressor.coef_, cnames, columns=['Coefficient'])
    print(coeff_df)

ejemplo_vino()