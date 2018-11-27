import pandas as pd #Data load
MissingData = pd.read_csv('missingdatas.csv')
country=MissingData.iloc[:,0:1].values #Country column selected
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le=LabelEncoder() #Object to assign numeric values to categoric datas
country[:,0]= le.fit_transform(country) #apply these changes to dataset.
ohe= OneHotEncoder(categorical_features='all')#provides to make zero the other attributes when one of them equals '1'
country=ohe.fit_transform(country).toarray() #apply the affect of onehotencoder object to dataset
Age= MissingData.iloc[:,1:4].values #apply the affect of onehotencoder object to dataset
result=pd.DataFrame(data=country,index=range(22),columns=['fr','tr','us']) #create a dataframe we encoded before from index 0 to 21 for country column
result2=pd.DataFrame(data=Age,index=range(22),columns=['height','weight','age']) #create a dataframe we encoded before from index 0 to 21 for height,weight and age columns
gender=MissingData.iloc[:,-1:].values #Selects the gender column
result3=pd.DataFrame(data=gender,index=range(22),columns=['gender']) #create a dataframe we encoded before from index 0 to 21 for height,weight and
s=pd.concat([result,result2],axis=1)#concatanation of result and result2
print(s)
