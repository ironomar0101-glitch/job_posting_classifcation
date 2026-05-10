import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

data=pd.read_csv(r'D:\ML\job posting_DS\fake_job_postings.csv')
data.head()
data.drop(columns='job_id',inplace=True)

def show_info(data):
    print('='*50)
    data.info()

    print('='*50)
    stats=data.describe().T
    print(stats)

    print('='*50)
    print(f'Rows:{data.shape[0]} Columns:{data.shape[1]}')

def show_duplicate(data):
    dup=data.duplicated().sum()
    print('='*50)
    print(dup)

def show_Nan(data):
    print('='*50)
    NaN=data.isna().sum()
    print(NaN)
    print('='*50)
    print(NaN/len(data))

def cat_col(data):
    print('='*50)
    col=[col for col in data.columns if data[col].dtypes=='object']
    for i in col:
        print(f'{i} has {data[i].nunique()} unique values: {data[i].unique()}')       

def num_col(data):
    print('='*50)
    col=[col for col in data.columns if data[col].dtypes !='object'] 
    print(col)   

def show_imbalance(data):
    plt.figure(figsize=(14,6))
    sns.countplot(x='fraudulent',data=data,palette='coolwarm',hue='fraudulent')
    plt.title('Imbalance Class')
    plt.xlabel('Fraudulent Classes')
    plt.grid()
    plt.show()

def show_class_distribution(data):
    plt.figure(figsize=(14,6))
    sns.histplot(x='fraudulent',data=data,palette='viridis',hue='fraudulent')
    plt.title('Distribution Of Classes')
    plt.xlabel('Fraudulent Classes')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

def experience_fraudulent(data):
    plt.figure(figsize=(14,6))
    sns.countplot(x='required_experience',data=data,hue='fraudulent',palette='viridis')
    plt.title('Relation between Experience and Fraudulent')
    plt.xlabel('Experience')
    plt.grid()
    plt.show()

def education_fraudulent(data):
    plt.figure(figsize=(30,15))
    sns.countplot(x='required_education',data=data,palette='viridis',hue='fraudulent')
    plt.title('Relation between Education and Fraudulent')
    plt.xlabel('Education')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

def employment_fraudulent(data):
    employment_df=data.groupby('employment_type')['fraudulent'].mean()
    fig = px.bar(
    x=employment_df.index,
    y=employment_df.values,
    labels={'x': 'Employment Type', 'y': 'Frequency'})

    fig.update_traces(
        text=employment_df.values,
        textposition='outside',
        marker_color='royalblue'
    )

    fig.update_layout(
        title={'text': "Employment Type Fraudulent Frequency", 'x': 0.5},
        xaxis_title="Employment Type",
        yaxis_title="Frequency",
        template="plotly_white"
    )

    fig.show()

def funcation_fraudulent(data):
    function_df=data.groupby('function')['fraudulent'].mean()
    fig=px.bar(x=function_df.index,y=function_df.values,
            labels={'x':'funcatio','y':'fraudulent'})
    fig.update_traces(
        text=function_df.values,
        textposition='outside',
        marker_color='royalblue'
    )

    fig.update_layout(
        title={'text': "function Type Fraudulent Frequency", 'x': 0.5},
        xaxis_title="function",
        yaxis_title="Frequency",
        template="plotly_white"
    )

    fig.show()

def correlation(data):
    plt.figure(figsize=(14,6))
    corr=data.corr(numeric_only=True)
    sns.heatmap(corr,annot=True,cmap='viridis')
    plt.title('person\'s r')
    plt.show()

def outlier(data):
    q1 = data.select_dtypes(exclude='object').quantile(0.25)
    q3 = data.select_dtypes(exclude='object').quantile(0.75)
    iqr = q3 - q1

    lb = q1 - 1.5*iqr
    ub = q3 + 1.5*iqr

    mask=(data.select_dtypes(exclude='object') < lb) | (data.select_dtypes(exclude='object')>ub)
    outlier=data[mask.any(axis=1)]

    print('='*50)
    print(outlier)

def Nlp_length(data):

    data['desc_len'] = data['description'].str.len()

    plt.figure(figsize=(14,6))
    sns.histplot(data=data, x='desc_len', hue='fraudulent', kde=True, palette='viridis')
    plt.title('Distribution of Description Length (Real vs Fraudulent)')
    plt.xlabel('Number of Characters')
    plt.show()


def show_all(data):
    show_info(data)
    show_duplicate(data)
    show_Nan(data)
    cat_col(data)
    num_col(data)
    show_imbalance(data)
    show_class_distribution(data)
    experience_fraudulent(data)
    education_fraudulent(data)
    employment_fraudulent(data)
    funcation_fraudulent(data)
    correlation(data)
    outlier(data)
    Nlp_length(data)
    
show_all(data)