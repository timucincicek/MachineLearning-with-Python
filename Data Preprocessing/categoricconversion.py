import pandas as pd 
MissingData = pd.read_csv('missingdatas.csv') #Data load
country=MissingData.iloc[:,0:1].values #Country column selected
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le=LabelEncoder() #Object to assign numeric values to categoric datas
country[:,0]= le.fit_transform(country) #apply these changes to dataset.
ohe= OneHotEncoder(categorical_features='all')#provides to make zero the other attributes when one of them equals '1'
country=ohe.fit_transform(country).toarray()#apply the affect of onehotencoder object to dataset
print(country)
