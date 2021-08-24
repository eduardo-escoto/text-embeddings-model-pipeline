import datetime as dt
from read_data import *
from process_data import *
from train_model import *
import yaml
import pymysql
import os



def get_dates(pipeline_config, prediction_only = False):
    date_dict = {}
    date_dict['prediction_date_end'] = dt.datetime.now().date() - dt.timedelta(days = pipeline_config['prediction_end_lag'])
    date_dict['prediction_date_start'] = date_dict['prediction_date_end'] - dt.timedelta(days = pipeline_config['n_prediction_days'])
    
    if prediction_only:
        return date_dict
    
    date_dict['validation_date_end'] = date_dict['prediction_date_start'] - dt.timedelta(days = pipeline_config['validation_end_lag'])
    date_dict['validation_date_start'] = date_dict['validation_date_end'] - dt.timedelta(days = pipeline_config['n_validation_days'])
    date_dict['train_date_end'] = date_dict['validation_date_start'] - dt.timedelta(days = pipeline_config['train_end_lag'])
    date_dict['train_date_start'] = date_dict['train_date_end'] - dt.timedelta(days = pipeline_config['n_train_days'])
    
    
    return date_dict
    
    
def get_model_info(model_info_config):
    model_info_conn = pymysql.connect(**model_info_config['db_args'])
    model_info_query = model_info_config['get_model_info_query'].format(**model_info_config)
    model_info_df = pd.read_sql(model_info_query, model_info_conn)
    model_info_conn.close()
    return model_info_df

def get_model_base_path(model_info_config):
    model_info = get_model_info(model_info_config)
    return model_info['model_output_base_dir'][0]


def is_pending_training(pipeline_config, model_info_config):
    today = dt.date.today()
    model_info_df = get_model_info(model_info_config)
    
    if pipeline_config['force_train']:
        return True
    if model_info_df.shape[0] == 0:
        return True
    if str(model_info_df['model_expires_on'][0]) == str(today):
        return True

    return False

def make_folder_structure(model_train_date):
    model_data_base_path = (
        os.getcwd() + '/models/' 
    )
    model_folder = "_".join(str(dt.datetime.now()).split(" "))
    model_full_path = model_data_base_path + model_folder
    
    if not os.path.isdir(model_data_base_path):
        os.mkdir(model_data_base_path)
    
    os.mkdir(model_full_path)
    os.mkdir(model_full_path + "/encoders")
    os.mkdir(model_full_path + "/scalers")
    os.mkdir(model_full_path + "/imputers")
    os.mkdir(model_full_path + "/tokenizers")
       
    return model_full_path

def update_metadata(metadata, model_info_config):
    model_data_update = model_info_config['insert_model_info_query'].format(**metadata)
    model_info_conn = pymysql.connect(**model_info_config['db_args'])
    
    cur = model_info_conn.cursor()
    cur.execute('SET autocommit = 1')
    cur.execute('set sql_safe_updates = 0')
    cur.execute(model_data_update)
    cur.close()
    model_info_conn.close()
    
    
def upload_preds(pred_data, process_config, model_info_config, pipeline_config):
    model_info_conn = pymysql.connect(**model_info_config['db_args'])
    cur = model_info_conn.cursor()
    cur.execute('SET autocommit = 1')
    cur.execute('set sql_safe_updates = 0')
    
    for index, row in pred_data.reset_index().iterrows():
        output_data = dict(zip(
            process_config['index_features'], 
            row[process_config['index_features']]
        ))
        output_data ['prediction'] = row['prediction']
        insert_query = pipeline_config['output_query'].format(
            **output_data,
            **pipeline_config
        )
        cur.execute(insert_query)
    cur.close()
    model_info_conn.close()

def train_and_predict(process_config, pipeline_config, model_info_config, model_config):
    model_train_date = str(dt.datetime.now())
    model_expires_date = str(
        (dt.datetime.now() + dt.timedelta(days = pipeline_config['model_train_interval'])
    ).date())
    # create path to save stuff in
    model_path = make_folder_structure(model_train_date)
    print(model_path)
    print('gathering data')
    # create dates
    process_config = {**process_config, **get_dates(pipeline_config)}

    # read data
    all_data = read_data(make_data_dict(process_config['train_date_start'], process_config['prediction_date_end']))
    
    print('processing data')
    # process all data and save imputers, etc
    train, val, test = process_data(all_data, process_config, model_path)
    print('training model')
    # train model
    model, out_train, out_val, out_test = run_model(train, val, test, model_path, process_config, model_config)

    print('validating model')
    # get validations
    val_data = get_validation_data(out_train, out_val, out_test)

    print('uploading model metrics')
    # get, and upload metadata
    metadata = {
        **model_info_config,
        **val_data,
        **process_config,
        'model_trained_on': model_train_date,
        'model_expires_on': model_expires_date,
        'model_output_base_dir': model_path
    }
    update_metadata(metadata, model_info_config)

    print('uploading predictions')
    # upload predictions
    upload_preds(out_test, process_config, model_info_config, pipeline_config)
    
def predict_only(process_config, pipeline_config, model_info_config, model_config):
    print("predicting only")
    # Get path to read models, processing, etc
    print('getting model path')
    model_path = get_model_base_path(model_info_config)
    # get prediction dates only
    print('reading data')
    process_config = {**process_config, **get_dates(pipeline_config, prediction_only = True)}
    # read prediction data
    pred_data = read_data(make_data_dict(process_config['prediction_date_start'], process_config['prediction_date_end']))
    # process predictions
    print('processing data')
    pred = process_data(pred_data, process_config, model_path, prediction_only = True)
    # load model and predict
    print('loading model and generating predictions')
    model, util = run_model_predict(pred, model_path, process_config, model_config)
    # upload predictions
    print('uploading predictions')
    upload_preds(util, process_config, model_info_config, pipeline_config)

def run_pipeline(process_config, pipeline_config, model_info_config, model_config):
    if is_pending_training(pipeline_config, model_info_config):
        train_and_predict(process_config, pipeline_config, model_info_config, model_config)
    else:
        predict_only(process_config, pipeline_config, model_info_config, model_config)
        
        
if __name__ == "__main__":
    model_info_config = None
    with open("model_info_table_config.yml", "r") as config_file:
        model_info_config = yaml.safe_load(config_file)

    model_config = None
    with open("train_model_config.yml", "r") as config_file:
        model_config = yaml.safe_load(config_file)

    pipeline_config = None
    with open("run_model_pipeline_config.yml", "r") as config_file:
        pipeline_config = yaml.safe_load(config_file)

    process_config = None
    with open("process_data_config.yml", "r") as config_file:
        process_config = yaml.safe_load(config_file)
    
    run_pipeline(process_config, pipeline_config, model_info_config, model_config)