import pandas as pd #Data load
MissingData = pd.read_csv('missingdatas.csv')
country=MissingData.iloc[:,0:1].values #Country column selected
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le=LabelEncoder() #Object to assign numeric values to categoric datas
country[:,0]= le.fit_transform(country) #apply these changes to dataset.
ohe= OneHotEncoder(categorical_features='all')#provides to make zero the other attributes when one of them equals '1'
country=ohe.fit_transform(country).toarray() #apply the affect of onehotencoder object to dataset
Age= MissingData.iloc[:,1:4].values #apply the affect of onehotencoder object to dataset
result=pd.DataFrame(data=country,index=range(22),columns=['fr','tr','us'])#create a dataframe we encoded before from index 0 to 21
result2=pd.DataFrame(data=Age,index=range(22),columns=['height','weight','age'])
gender=MissingData.iloc[:,-1:].values
result3=pd.DataFrame(data=gender,index=range(22),columns=['gender'])
s=pd.concat([result,result2],axis=1)#concatanation of result and result2
print(s)
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s,result3,test_size=0.33,random_state=0) #Train and Test sets splitted.Machine will be trained and tested by x and y.We expect that machine will get the gender attribute according to training.Test is 1/3 and Training is 2/3 of all dataset.Random_state randomizes the datas to use datas from all countries. 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler() #To use standardization
X_train = sc.fit_transform(x_train) #Applied standardization for x_train
X_test = sc.fit_transform(x_test)   #Applied standardization for x_test
