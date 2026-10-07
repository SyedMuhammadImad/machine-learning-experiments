"""Portable numeric and tabular experiments; raw input and fitted artifacts stay local."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer,load_diabetes,load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,mean_squared_error,r2_score,adjusted_rand_score
from numerical import *

def classification_metrics(y,p):
    return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro')),'confusion_matrix':confusion_matrix(y,p).tolist()}

def tabular(data,target,drop=()):
    if target not in data or not len(data) or len(data)>100000:raise ValueError('Invalid target or dataset size')
    if data[target].isna().any():raise ValueError('Missing target labels')
    columns=[c for c in data if c!=target and c not in drop]
    if not columns:raise ValueError('No predictors remain')
    frame=data[columns+[target]].copy();frame=frame.replace([np.inf,-np.inf],np.nan)
    before=len(frame);group_columns=columns
    # Inconsistent labels for identical predictors are excluded before splitting.
    keys=pd.util.hash_pandas_object(frame[group_columns],index=False)
    conflicts=frame.assign(_key=keys).groupby('_key')[target].transform('nunique')>1
    frame=frame.loc[~conflicts].drop_duplicates();y=frame.pop(target).astype(str)
    if y.nunique()<2 or y.value_counts().min()<10:raise ValueError('At least two classes and ten unique rows per class required')
    xt,rest,yt,yr=train_test_split(frame,y,test_size=.4,random_state=42,stratify=y)
    xv,xe,yv,ye=train_test_split(rest,yr,test_size=.5,random_state=43,stratify=yr)
    numeric=xt.select_dtypes(include=['number']).columns.tolist();categorical=[c for c in xt if c not in numeric]
    pre=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),
        ('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),categorical)])
    candidates={'decision_tree':[DecisionTreeClassifier(max_depth=d,random_state=42) for d in [3,6,None]],
      'random_forest':[RandomForestClassifier(n_estimators=100,max_depth=d,random_state=42,n_jobs=2) for d in [6,None]],
      'knn':[KNeighborsClassifier(k) for k in [3,5,7]]}
    results={}
    for name,models in candidates.items():
        options=[]
        for model in models:
            from sklearn.base import clone
            pipeline=Pipeline([('prepare',clone(pre)),('model',model)]).fit(xt,yt)
            score=float(f1_score(yv,pipeline.predict(xv),average='macro'));options.append((score,pipeline))
        best=max(options,key=lambda option:option[0]);prediction=best[1].predict(xe)
        results[name]={'validation_macro_f1':best[0],'selected_settings':{k:v for k,v in best[1]['model'].get_params().items() if k in ['max_depth','n_neighbors','n_estimators']},'test':classification_metrics(ye,prediction)}
    return {'input_rows':before,'retained_rows':len(frame),'conflicting_rows_removed':int(conflicts.sum()),'duplicate_rows_removed':before-int(conflicts.sum())-len(frame),
        'train':len(xt),'validation':len(xv),'test':len(xe),'dropped_features':list(drop),'labels':sorted(y.unique().tolist()),'models':results,
        'selection':'validation macro F1; imputation, encoding and scaling fit training rows only',
        'limitation':'single random row split; time/source/entity separation and operational use are unverified'}

def balanced(labels,count,seed):
    rng=np.random.default_rng(seed);indices=[]
    for label in np.unique(labels):indices.extend(rng.choice(np.flatnonzero(labels==label),count//len(np.unique(labels)),replace=False))
    rng.shuffle(indices);return np.asarray(indices)

def mnist(data_dir):
    from torchvision.datasets import MNIST
    train=MNIST(data_dir,train=True,download=False);test=MNIST(data_dir,train=False,download=False)
    ti=balanced(train.targets.numpy(),6000,42);ei=balanced(test.targets.numpy(),1000,43)
    # Correctly threshold BOTH splits; never create test data from an already sliced train array.
    raw=(train.data.numpy()[ti].reshape(-1,784)>=127).astype(np.uint8);y=train.targets.numpy()[ti]
    x,v,yt,yv=train_test_split(raw,y,test_size=.2,stratify=y,random_state=44)
    tx=(test.data.numpy()[ei].reshape(-1,784)>=127).astype(np.uint8);ty=test.targets.numpy()[ei]
    choices=[]
    for alpha in [.1,.5,1.,2.]:
        model=BernoulliClassifier(alpha).fit(x,yt);score=float(f1_score(yv,model.predict_batch(v),average='macro'));choices.append((score,alpha,model))
    score,alpha,model=max(choices,key=lambda option:(option[0],option[1]));prediction=model.predict_batch(tx)
    reference=BernoulliNB(alpha=alpha,binarize=None).fit(x,yt)
    np.testing.assert_allclose(model.joint_log_likelihood(tx),reference._joint_log_likelihood(tx),atol=1e-10)
    np.testing.assert_array_equal(prediction,reference.predict(tx))
    return {'training':len(x),'validation':len(v),'official_test':len(tx),'threshold':127,'alpha':alpha,'validation_macro_f1':score,
        'manual_test':classification_metrics(ty,prediction),'library_test':classification_metrics(ty,reference.predict(tx)),
        'independent_likelihood_and_prediction_parity':True,'data_fingerprint':hashlib.sha256(raw.tobytes()+y.tobytes()+tx.tobytes()+ty.tobytes()).hexdigest(),
        'limitation':'balanced subsets and one seed; not full MNIST benchmark or real handwriting UI validation'}

def regression():
    data=load_diabetes();x=data.data[:,2:3];y=data.target
    xt,rest,yt,yr=train_test_split(x,y,test_size=.4,random_state=42)
    xv,xe,yv,ye=train_test_split(rest,yr,test_size=.5,random_state=43)
    slope,intercept=ordinary_least_squares(xt[:,0],yt);prediction=xe[:,0]*slope+intercept
    reference=LinearRegression().fit(xt,yt);np.testing.assert_allclose(prediction,reference.predict(xe),atol=1e-10)
    choices=[]
    for degree in range(1,6):
        model=Pipeline([('scale',StandardScaler()),('poly',PolynomialFeatures(degree,include_bias=False)),('regression',LinearRegression())]).fit(xt,yt)
        choices.append((float(mean_squared_error(yv,model.predict(xv))),degree,model))
    mse,degree,model=min(choices,key=lambda value:(value[0],value[1]));pred=model.predict(xe)
    return {'dataset':'bundled diabetes progression, one BMI predictor; a substitute for unavailable historic X/R CSVs',
        'train':len(xt),'validation':len(xv),'test':len(xe),'manual_linear':{'mse':float(mean_squared_error(ye,prediction)),'r2':float(r2_score(ye,prediction)),'library_prediction_matches':True},
        'polynomial':{'selected_degree':degree,'validation_mse':mse,'test_mse':float(mean_squared_error(ye,pred)),'test_r2':float(r2_score(ye,pred))},
        'limitation':'offline one-predictor benchmark; not clinical prediction or medical guidance'}

def clustering():
    iris=load_iris();x=StandardScaler().fit_transform(iris.data)
    manual=LloydKMeans(3).fit(x);library=KMeans(3,n_init=10,random_state=42).fit(x)
    return {'dataset':'150 Iris rows; unsupervised three-cluster experiment, not held-out classification',
        'manual':{'inertia':manual.inertia_,'adjusted_rand_index':float(adjusted_rand_score(iris.target,manual.labels_)),'iterations':len(manual.inertia_history)},
        'library':{'inertia':float(library.inertia_),'adjusted_rand_index':float(adjusted_rand_score(iris.target,library.labels_))}}

def table_operations(wine_file,demographic_file):
    wine=pd.read_json(wine_file);demographic=pd.read_csv(demographic_file);demographic.columns=demographic.columns.str.strip()
    required={'country','province','region_1','region_2','variety','points','description'}
    if not required<=set(wine):raise ValueError('Wine table missing required columns')
    if not {'City','Age','Gender','Family Income'}<=set(demographic):raise ValueError('Demographic table missing required columns')
    first=wine.iloc[[0,1,10,100]][['country','province','region_1','region_2']]
    high=wine.loc[wine.country.isin(['Australia','New Zealand']) & (wine.points>=95)]
    age=pd.to_numeric(demographic.Age,errors='raise');income=pd.to_numeric(demographic['Family Income'],errors='raise')
    # Aggregate results only: no names, addresses or individual records are exported.
    return {'wine_rows':len(wine),'italian_rows':int(wine.country.eq('Italy').sum()),'high_scoring_oceania_rows':len(high),
        'selected_location_shape':list(first.shape),'first_100_country_variety_shape':list(wine.iloc[:100][['country','variety']].shape),
        'missing_counts':{k:int(wine[k].isna().sum()) for k in ['country','province','region_1','region_2']},
        'demographic_rows':len(demographic),'nonmissing_age_rows':int(age.notna().sum()),'nonmissing_income_rows':int(income.notna().sum()),
        'mean_age':float(age.mean()),'median_income':float(income.median())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--mnist-dir',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.data_dir)
    try:
        datasets={'bank_contact_response':(pd.read_csv(root/'bank.csv'),'deposit',['duration']),
          'bank_personal_loan':(pd.read_csv(root/'UniversalBank.csv'),'Personal Loan',['ID','ZIP Code']),
          'titanic_survival':(pd.read_csv(root/'titanic-train.csv'),'Survived',['PassengerId','Name','Ticket','Cabin']),
          'car_acceptability':(pd.read_csv(root/'car.data',names=['buying','maint','doors','persons','lug_boot','safety','class']),'class',[])}
        breast=load_breast_cancer(as_frame=True);frame=breast.data.copy();frame['label']=breast.target
        datasets['breast_cell_benchmark']=(frame,'label',[])
        results={}
        for name,(frame,target,drop) in datasets.items():
            results[name]=tabular(frame,target,drop);print('Completed '+name,flush=True)
        result={'tabular':results,'bernoulli_mnist':mnist(args.mnist_dir),'regression':regression(),'clustering':clustering(),
            'table_operations':table_operations(root/'winemag-data-130k-v2.json',root/'demographics.csv'),
            'limitations':'No reproduction claim for missing Social Network Ads, Drug200, HR/credit, historic regression, image or plotting/cleaning-source datasets.'}
        Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        print('All real-input experiment branches completed.',flush=True)
    except (ValueError,OSError,KeyError) as error:p.exit(2,'Failed: '+str(error)+'\n')
if __name__=='__main__':main()
