import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

data = pd.read_csv("LoanApprovalPrediction.csv")  
print(data.head(5))


data.drop(['Loan_ID'],axis=1,inplace=True)
print(data.head(5))


#barplot
obj = (data.dtypes == 'object') 
object_cols = list(obj[obj].index) 
plt.figure(figsize=(18,36)) 
index = 1
  
for col in object_cols: 
  y = data[col].value_counts() 
  plt.subplot(11,4,index) 
  plt.xticks(rotation=0)
  sns.barplot(x=list(y.index), y=y) 
  index +=1

plt.show()


# Import label encoder 
from sklearn import preprocessing 
    
# label_encoder object knows how  
# to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
obj = (data.dtypes == 'object') 
for col in list(obj[obj].index): 
  data[col] = label_encoder.fit_transform(data[col])

print(data.head(5))




# plt.figure(figsize=(12,6)) 
  
# sns.heatmap(data.corr(),cmap='tab10',fmt='.2f', 
#             linewidths=2,annot=True)

# plt.show()



# sns.catplot(x="Gender", y="Married", 
#             hue="Loan_Status",  
#             kind="bar",  
#             data=data)

#plt.show()



for col in data.columns: 
  data[col] = data[col].fillna(data[col].mean()) 


from sklearn.model_selection import train_test_split 
  
X = data.drop(['Loan_Status'],axis=1) 
Y = data['Loan_Status'] 
X.shape,Y.shape 

print(X.shape, Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                    test_size=0.4, 
                                                    random_state=1) 

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)







from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 
  
from sklearn import metrics 
  
knn = KNeighborsClassifier(n_neighbors=3) 
rfc = RandomForestClassifier(n_estimators = 7, 
                             criterion = 'entropy', 
                             random_state =7) 
svc = SVC() 
lc = LogisticRegression() 
  
# making predictions on the training set 
for clf in (rfc, knn, svc,lc): 
    clf.fit(X_train, Y_train) 
    Y_pred = clf.predict(X_test) 
    print("Accuracy score of ", 
          clf.__class__.__name__, 
          "=",100*metrics.accuracy_score(Y_test,  
                                         Y_pred))
    #confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_test, Y_pred)
    print(cm)   
    sns.heatmap(cm, annot=True)
    plt.show()

    #f1 score
    from sklearn.metrics import f1_score
    print(f1_score(Y_test, Y_pred, average='weighted'))
    


