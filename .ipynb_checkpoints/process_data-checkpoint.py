import pandas as pd
import numpy as np
import pickle
import yaml
import nltk

from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus  import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def find_non_rare_labels(df, variable, tolerance):
    temp = df.groupby([variable])[variable].count() / len(df)
    non_rare = [x for x in temp.loc[temp>tolerance].index.values]
    return non_rare


def remove_dates(df, params):
    df = df.loc[-df[params['date_feature']].astype('str').isin(params['excluded_dates']), :]
    return df

def split_by_date(df, params, drop_date = True):  
    
    if 'test_date_start' not in params.keys() and 'prediction_date_start' in params.keys():
        params['test_date_start'] = params['prediction_date_start']
    if 'test_date_end' not in params.keys() and 'prediction_date_end' in params.keys():
        params['test_date_end'] = params['prediction_date_end']
    
    
    df_train = df[
        (df[params['date_feature']] >= str(params['train_date_start']))
        & (df[params['date_feature']] <= str(params['train_date_end']))
    ].copy()
    
    df_val = df[
        (df[params['date_feature']] >= str(params['validation_date_start']))
        & (df[params['date_feature']] <= str(params['validation_date_end']))
    ].copy()
    
    df_test = df[
        (df[params['date_feature']] >= str(params['test_date_start']))
        & (df[params['date_feature']] <= str(params['test_date_end']))
    ].copy()
    
    if drop_date:
        df_train.drop(params['date_feature'], axis = 1, inplace = True)
        df_val.drop(params['date_feature'], axis = 1, inplace = True)
        df_test.drop(params['date_feature'], axis = 1, inplace = True)
        
    return df_train, df_val, df_test

def append_smalls(categories):
    if 'smalls' not in categories:
        categories = np.array([*categories, 'smalls'], dtype = categories.dtype)
    return categories
    
def process_num_preds(num, params, model_path):
    sc = None
    with open(model_path + '/scalers/standard_scaler.pickle', 'rb') as handle:
        sc = pickle.load(handle)
    num = num.replace('NA', np.nan)
    num = num.replace('', np.nan)
    num.drop(params['date_feature'], axis = 1, inplace = True)
    for col in num.columns:
        if col != params['date_feature']:
            try:
                mode = None
                with open(model_path + '/imputers/mode_' + col +'.pickle', 'rb') as handle:
                    mode = pickle.load(handle)

                num[col] = num[col].fillna(mode) 
                num[col] = num[col].astype(np.float32)
            except:
                num.drop([col], axis = 1, inplace = True)
                print('{} dropped!'.format(col))
    
    
    num_cols = num.columns
    df_num = sc.transform(num)
    df_num = pd.DataFrame(df_num, columns = num_cols, index = num.index)
    
    return df_num

def process_num(num, params, model_path):
    sc = StandardScaler()
    
    num = num.replace('NA', np.nan)
    num = num.replace('', np.nan)
    
    for col in num.columns:
        if col != params['date_feature']:
            try:
                num[col] = num[col].astype(np.float32)
                mode = num[col].mode()[0]
                num[col] = num[col].fillna(mode)
                with open(model_path + '/imputers/mode_' + col +'.pickle', 'wb') as handle:
                    pickle.dump(mode, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            except:
                num.drop([col], axis = 1, inplace = True)
                print('{} dropped!'.format(col))
    
    
    num_train, num_val, num_test = split_by_date(num, params)
    num_cols = num_train.columns
    
    sc.fit(num_train)

    df_train_num = sc.transform(num_train)
    df_val_num = sc.transform(num_val)
    df_test_num = sc.transform(num_test)

    df_train_num = pd.DataFrame(df_train_num, columns = num_cols, index = num_train.index)
    df_val_num = pd.DataFrame(df_val_num, columns = num_cols, index = num_val.index)
    df_test_num = pd.DataFrame(df_test_num, columns = num_cols, index = num_test.index)
    
    with open(model_path + '/scalers/standard_scaler.pickle', 'wb') as handle:
        pickle.dump(sc, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    return df_train_num, df_val_num, df_test_num

def process_cat_preds(cat, params, model_path):
    ohe = None
    with open(model_path + '/encoders/one_hot_encoder.pickle', 'rb') as handle:
        ohe = pickle.load(handle)
    
    cat = cat.replace('NA', np.nan)
    cat = cat.replace('', np.nan)
    cat = cat.replace(np.nan, 'NL')
    cat.drop(params['date_feature'], axis = 1, inplace = True)
    
    for col in cat.columns:
        if col != params['date_feature']:
            frequent_cat = None
            with open(model_path + '/imputers/smalls_' + col +'.pickle', 'rb') as handle:
                frequent_cat = pickle.load(handle)
                
            cat[col] = np.where(cat[col].isin(frequent_cat), cat[col], 'smalls')
        
    cat_cols = cat.columns
    ohe_cols = ohe.get_feature_names(cat_cols.values)
    
    df_cat = ohe.transform(cat.values)
    df_cat = pd.DataFrame(df_cat, columns = ohe_cols, index = cat.index)
    
    return df_cat


def process_cat(cat, params, model_path):
    ohe = OneHotEncoder(sparse = False)
    
    cat = cat.replace('NA', np.nan)
    cat = cat.replace('', np.nan)
    cat = cat.replace(np.nan, 'NL')
    
    for col in cat.columns:
        if col != params['date_feature']:
            frequent_cat = find_non_rare_labels(cat, col, params['smalls'])
            cat[col] = np.where(cat[col].isin(frequent_cat), cat[col], 'smalls')
            
            with open(model_path + '/imputers/smalls_' + col +'.pickle', 'wb') as handle:
                pickle.dump(frequent_cat, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    cat_train, cat_val, cat_test = split_by_date(cat, params)
    cat_cols = cat_train.columns
    
    ohe.fit(cat_train)
    ohe.categories_ =  [append_smalls(cat) for cat in ohe.categories_]
    
    ohe_cols = ohe.get_feature_names(cat_cols.values)
    
    df_train_cat = ohe.transform(cat_train.values)
    df_val_cat = ohe.transform(cat_val.values)
    df_test_cat = ohe.transform(cat_test.values)
    
    df_train_cat = pd.DataFrame(df_train_cat, columns = ohe_cols, index = cat_train.index)
    df_val_cat = pd.DataFrame(df_val_cat, columns = ohe_cols, index = cat_val.index)
    df_test_cat = pd.DataFrame(df_test_cat, columns = ohe_cols, index = cat_test.index)
    
    with open(model_path + '/encoders/one_hot_encoder.pickle', 'wb') as handle:
        pickle.dump(ohe, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return df_train_cat, df_val_cat, df_test_cat

def clean_text(text_input):
    cleaner = RegexpTokenizer(r'\w+')
    stop_words = stopwords.words('english')
    stemmer = PorterStemmer()
    text = [x.lower().strip() for x in text_input]
    tokenized = [word_tokenize(x) for x in text]
    tokenized = [(filter(None, x)) for x in tokenized]
    cleaned_tokenized = [cleaner.tokenize(" ".join(x)) for x in tokenized]
    filtered_tokenized = [[s for s in x if s not in stop_words] for x in cleaned_tokenized]
    sentences = [" ".join(row) for row in filtered_tokenized]

    return pd.Series(data = sentences, index = text_input.index)

def process_nlp(nlp, params, prediction_only = False):
    for col in nlp.columns:
        if col != params['date_feature']:
            nlp[col] = clean_text(nlp[col])
            
    if prediction_only:
        return nlp
    
    nlp_train, nlp_val, nlp_test = split_by_date(nlp, params)
    
    return nlp_train, nlp_val, nlp_test

def process_data_train(in_df, params, model_path):
    df = in_df.copy()
    df.set_index(params['index_features'], inplace = True)
    df[params['date_feature']] = pd.to_datetime(df[params['date_feature']])
    df[params['target_feature']] = df[params['target_feature']].astype(np.float64)
    
    df = remove_dates(df, params)
    
    util = df[params['utility_features'] + [params['target_feature'], params['date_feature']]].copy()
    nlp = df[params['nlp_features'] + [params['date_feature']]].copy()
    cat = df[params['categoric_features'] + [params['date_feature']]].copy()
    num = df[params['numeric_features'] + [params['date_feature']]].copy()
    
    
    cat_train, cat_val, cat_test = process_cat(cat, params, model_path)
    num_train, num_val, num_test = process_num(num, params, model_path)   
    nlp_train, nlp_val, nlp_test = process_nlp(nlp, params)
    util_train, util_val, util_test = split_by_date(util, params, drop_date = False)
    
    train = pd.concat([util_train, cat_train, nlp_train, num_train], axis = 1)
    val = pd.concat([util_val, cat_val, nlp_val, num_val], axis = 1)
    test = pd.concat([util_test, cat_test, nlp_test, num_test], axis = 1)
    
    return train, val, test 

def process_data_predict(in_df, params, model_path):
    df = in_df.copy()
    df.set_index(params['index_features'], inplace = True)
    df[params['date_feature']] = pd.to_datetime(df[params['date_feature']])
    df[params['target_feature']] = df[params['target_feature']].astype(np.float64)
    
    util = df[params['utility_features'] + [params['target_feature'], params['date_feature']]].copy()
    nlp = df[params['nlp_features'] + [params['date_feature']]].copy()
    cat = df[params['categoric_features'] + [params['date_feature']]].copy()
    num = df[params['numeric_features'] + [params['date_feature']]].copy()
    
    
    cat_pred = process_cat_preds(cat, params, model_path)
    num_pred = process_num_preds(num, params, model_path)   
    nlp_pred = process_nlp(nlp, params, prediction_only = True)
    util_pred = util
    
    pred = pd.concat([util_pred, cat_pred, nlp_pred, num_pred], axis = 1)
    
    return pred

def process_data(in_df, params, model_path, prediction_only = False):
    nltk.data.path.append(params['nltk_data_dir'])
    if prediction_only:
        return process_data_predict(in_df, params, model_path)
    
    return process_data_train(in_df, params, model_path)