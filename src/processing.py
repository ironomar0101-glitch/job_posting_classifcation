import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(r'D:\ML\job posting_DS\fake_job_postings.csv')
data=data.drop(columns='job_id')

def fix_salary(data,col):
    values=data[data[col].notna()].loc[:,col].value_counts()
    print('='*50)
    print(values)

    data[f'{col}_missed']=np.where(data[col].isna(),1,0)
    data=data.drop(columns=col)

    return data

def fix_location(data,col):
    values=data[col].value_counts(sort=True).head(50)
    print('='*50)
    print(values)

    data['country']=data[col].str.extract(r'([A-Z]+)')
    data['state']=data[col].str.extract(r',\s*([A-Z]+)')
    data['city'] = data[col].str.extract(r',\s*([a-zA-Z\s]+)$')
    data['city']=data['city'].replace(' ',np.nan)

    data['country']=data['country'].fillna('unknown')
    data['state']=data['state'].fillna('unknown')
    data['city']=data['city'].fillna('unknown')

    data=data.drop(columns=col)
    return data

def fix_unknown(data, unkown_col):

    for col in unkown_col:
        if col in data.columns:
            data[f'{col}_missed'] = np.where(data[col].isna(), 1, 0)
            data[col] = data[col].fillna('unknown')
            
    return data

unkown_col= ['department', 'company_profile', 'requirements', 'benefits','industry']
other_col=['function','employment_type']

def fix_other(data,other_col):
    for col in other_col:
        if col in data.columns:
            data[col]=data[col].replace(np.nan,'Other')
    return data

def fix_experience(data,col):
    data[col]=data[col].replace(np.nan,'Not Applicable')
    return data

def fix_edu(data,col):
    data[col]=data[col].replace(np.nan,'Unspecified')
    return data

def drop_nan(data):
    data=data.dropna()
    return data

def drop_dup(data):
    dup=data.duplicated()
    print('='*50)
    print(data[dup])
    data=data.drop_duplicates()

    return data

def final_corr(data):
    plt.figure(figsize=(14,6))
    corr=data.corr(numeric_only=True)
    sns.heatmap(corr,annot=True,cmap='viridis')
    plt.title('person\'s r')

required_experience_dic={
    'Not Applicable':-1,
    'Internship':0,
    'Entry level':1,
    'Associate':2,
    'Mid-Senior level':3,
    'Director':4,
    'Executive':5,
}

required_education_dic={
    'Unspecified':-1,
    'Some High School Coursework':0,
    'High School or equivalent':1,
    'Vocational':2,
    'Vocational - HS Diploma':3,
    'Vocational - Degree':4,
    'Certification':5,
    'Some College Coursework Completed':6,
    'Associate Degree':7,
    "Bachelor's Degree":8,
    'Professional':9,
    "Master's Degree":10,
    'Doctorate':11
}
ordinal_col=['required_experience','required_education',]
def encode_ordinal(data,ordinal_col,dic):
    for col in ordinal_col:
        if col in data.columns:
            data[col]=data[col].map(dic)

    return data        

def encode_dep(data,col):
    freq=data[col].value_counts()
    top=freq[freq>10].index
    print('='*50)
    print(top)

    data[col]=np.where(data[col].isin(top),data[col],'other')
    unique=data[col].unique()
    print('='*50)
    print(unique)

    data[col]=data[col].str.lower().str.strip()
    unique=data[col].unique()
    print('='*50)
    print(unique)

    data[col]=data[col].replace('[^a-zA-Z]','',regex=True)
    unique=data[col].unique()
    print('='*50)
    print(unique)

    return data

dep_dic={
    'hr':'human resources',
    'it':'information technology',
    'admin':'administrative'
}

def replace_dep(data,col,dic):
    data[col]=data[col].replace(dic)
    unique=data[col].unique()
    print('='*50)
    print(unique)

    return(data)

def fix_fun(data,col):
    data[col]=data[col].str.lower().str.strip()
    data[col]=data[col].replace(r'[^a-zA-Z\s]','',regex=True)
    unique=data[col].unique()
    
    print('='*50)
    print(unique)
    return data

def fix_industry(data,col):
    data[col]=data[col].str.lower().str.strip()
    data[col]=data[col].replace(r'[^a-zA-Z\s]','',regex=True)
    unique=data[col].unique()
    
    print('='*50)
    print(unique)

    freq=data[col].value_counts()
    count=freq[freq>20].index
    
    print('='*50)
    print(len(count))

    data[col]=np.where(data[col].isin(count),data[col],'other')

    return data

location=['country','state','city']
def fix_address(data,location):
    for col in location:
        if col in data.columns:
            freq=data[col].value_counts()
            count=freq[freq>20].index

            print('='*50)
            print(len(count))

            data[col]=np.where(data[col].isin(count),data[col],'other')
    return data

def fix_city(data,col):
    data[col]=data[col].str.replace('New York City','New York')
    return data

nominal_col=['department','employment_type','industry','function','country','state','city']
def dummy(data,nominal_col):
    data=pd.get_dummies(columns=nominal_col,data=data,drop_first=True,dtype=int)    
    return data


def all_process(data):
    data=fix_salary(data,'salary_range')
    data=fix_location(data,'location')
    data=fix_unknown(data, unkown_col)
    data=fix_other(data,other_col)
    data=fix_experience(data,'required_experience')
    data=fix_edu(data,'required_education')
    data=drop_nan(data)
    data=drop_dup(data)
    final_corr(data)
    data=encode_ordinal(data,['required_experience'],required_experience_dic)
    data=encode_ordinal(data,['required_education']	,required_education_dic)
    data=encode_dep(data,'department')
    data=replace_dep(data,'department',dep_dic)
    data=fix_fun(data,'function')
    data=fix_industry(data,'industry')
    data=fix_city(data,'city')
    data=fix_address(data,location)
    data=dummy(data,nominal_col)
    return data

data_copy=data.copy()
data_copy=all_process(data_copy)
print(data_copy)