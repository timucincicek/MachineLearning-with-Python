import pandas as pd #Data load

MissingData = pd.read_csv('missingdatas.csv')
#sklearn comes from sci-kit learn
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy='mean',axis=0)
Age= MissingData.iloc[:,1:4].values #Choosed only numeric data columns
imputer = imputer.fit(Age[:,1:4]) #strategy applied for each column
Age[:,1:4]=imputer.transform(Age[:,1:4]) # Nan datas has been changed
print(Age)
