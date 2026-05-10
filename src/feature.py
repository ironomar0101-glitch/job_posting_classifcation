import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag

import re
from processing import all_process

lemma=WordNetLemmatizer()

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

def clean_corpus(text):
    text=re.sub('[^a-zA-Z]',' ',text)
    text=text.lower()
    text=word_tokenize(text)
    pos=pos_tag(text)
    corpus=[]
    for word,tag in pos:
        if word not in stop:
            char=tag[0].lower()
            p_tag=char if char in ['a','r','n','v'] else 'n'
            wn=lemma.lemmatize(word,pos=p_tag)
            corpus.append(wn)

    return ' '.join(corpus)    

df['text']=df['text'].apply(clean_corpus)  
print('='*50)  
print(df['text'][0])

def drop_text(data,text_cols):
    data= data.drop(columns=text_cols)
    return data

df=drop_text(df,text_cols)
df.head()
df = df.reset_index(drop=True)
x=df.drop(columns='fraudulent')
y=df['fraudulent']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
print(f'x_train_shape:{x_train.shape[0]}')
print(f'x_test_shape:{x_test.shape[0]}')
print(f'y_train_shape:{y_train.shape[0]}')
print(f'y_test_shape:{y_test.shape[0]}')

from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer(max_features=10000,ngram_range=(1,3))

def vectorizer(tf,x_train,x_test):
    tf_train=tf.fit_transform(x_train['text'])
    tf_test=tf.transform(x_test['text'])
    return tf_train,tf_test

tf_train,tf_test=vectorizer(tf,x_train,x_test)

x_train_=x_train.drop(columns='text')
x_test_=x_test.drop(columns='text')

from scipy.sparse import hstack
x_train_f=hstack((x_train_.values,tf_train))
x_test_f=hstack((x_test_.values,tf_test))

from imblearn.over_sampling import SMOTE
over=SMOTE(random_state=42)

def balance(over,x_train,y_train):
    x_train_res,y_train_res=over.fit_resample(x_train,y_train)
    return x_train_res,y_train_res

x_train_res,y_train_res=balance(over,x_train_f,y_train)

from sklearn.preprocessing import RobustScaler
scale_model=RobustScaler(with_centering=False)

def scale(model,x_train,x_test):

    x_train_scale=model.fit_transform(x_train)
    x_test_scale=model.transform(x_test)
    return x_train_scale,x_test_scale

x_train_scale,x_test_scale=scale(scale_model,x_train_res,x_test_f)