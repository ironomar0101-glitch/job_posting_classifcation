import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag

import re 
from processing import all_process
from feature import clean_corpus,drop_text,vectorizer,balance,scale

data=pd.read_csv(r'D:\ML\job posting_DS\fake_job_postings.csv')
df=all_process(data)
print('='*50)
df.head()

stop=stopwords.words('english')
print('='*50)
print(stop)

text_cols=df.select_dtypes(include='object').columns
print('='*50)
print(text_cols)

df['text']=df[text_cols].agg(' '.join , axis=1)
print('='*50)
df['text']

df['text']=df['text'].apply(clean_corpus)  
print('='*50)  
print(df['text'][0])

df=drop_text(df,text_cols)
df.head()
df = df.reset_index(drop=True)
x=df.drop(columns='fraudulent')
y=df['fraudulent']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)

from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer(max_features=10000,ngram_range=(1,3))
tf_train,tf_test=vectorizer(tf,x_train,x_test)

x_train_=x_train.drop(columns='text')
x_test_=x_test.drop(columns='text')

from scipy.sparse import hstack
x_train_f=hstack((x_train_.values,tf_train))
x_test_f=hstack((x_test_.values,tf_test))

from imblearn.over_sampling import SMOTE
over=SMOTE(random_state=42)
x_train_res,y_train_res=balance(over,x_train_f,y_train)

from sklearn.preprocessing import RobustScaler
scale_model=RobustScaler(with_centering=False)
x_train_scale,x_test_scale=scale(scale_model,x_train_res,x_test_f)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

models_1={
    'rf':RandomForestClassifier(),
    'dt':DecisionTreeClassifier(),
}
scores_1={}

for key_1,model_1 in models_1.items():
    score_1=cross_val_score(model_1,x_train_scale,y_train_res,cv=3,scoring='accuracy',n_jobs=-1)
    scores_1[key_1]=score_1.mean()

sort_score_1=sorted(scores_1.items(),key=lambda x :x[1],reverse=True)    
for name_1 , score_1 in sort_score_1:
    print(f'{name_1}:{score_1}')

models_2={
    'lr':LogisticRegression(),
    'ada':AdaBoostClassifier(),
    'knn':KNeighborsClassifier()
}
scores_2={}

for key_2,model_2 in models_2.items():
    score_2=cross_val_score(model_2,x_train_scale,y_train_res,cv=3,scoring='accuracy',n_jobs=-1)
    scores_2[key_2]=score_2.mean()

sort_score_2=sorted(scores_2.items(),key=lambda x :x[1],reverse=True)    
for name_2 , score_2 in sort_score_2:
    print(f'{name_2}:{score_2}')

models_3={
 
    'gb':GradientBoostingClassifier(),
}
scores_3={}

for key_3,model_3 in models_3.items():
    score_3=cross_val_score(model_3,x_train_scale,y_train_res,cv=3,scoring='accuracy',n_jobs=-1)
    scores_3[key_3]=score_3.mean()

sort_score_3=sorted(scores_3.items(),key=lambda x :x[1],reverse=True)    
for name_3 , score_3 in sort_score_3:
    print(f'{name_3}:{score_3}')

models_4={
    'svm':SVC(),
}
scores_4={}

for key_4,model_4 in models_4.items():
    score=cross_val_score(model_4,x_train_scale,y_train_res,cv=3,scoring='accuracy',n_jobs=-1)
    scores_4[key_4]=score.mean()

sort_score_4=sorted(scores_4.items(),key=lambda x :x[1],reverse=True)    
for name_4 , score_4 in sort_score_4:
    print(f'{name_4}:{score_4}')

from sklearn.model_selection import GridSearchCV
model=RandomForestClassifier()
param={
        'n_estimators':[100,150,200],
        'max_depth':[10,15,20],
        'min_samples_leaf':[3,5,10,15]
       }

gcv=GridSearchCV(estimator=model,param_grid=param,cv=5,scoring='accuracy')
gcv.fit(x_train_scale,y_train_res)
print(f'best estimator:{gcv.best_estimator_}')

pred=gcv.predict(x_test_scale)
print(pred)

from sklearn.metrics import classification_report
cr=classification_report(pred,y_test)
print(cr)

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
cm=confusion_matrix(pred,y_test)
plt.figure(figsize=(14,6))
ConfusionMatrixDisplay(cm).plot()
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import roc_auc_score,roc_curve
prob=gcv.predict_proba(x_test_scale)[:,1]
roc=roc_auc_score(y_test,prob)
print(roc)

fpr,tpr,threshold=roc_curve(y_test,prob)

plt.plot(fpr,tpr)
plt.plot([0,1],[0,1],linestyle='--')
plt.title('roc_curve')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()