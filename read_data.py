import yaml
import pymysql
import time
import datetime as dt
import numpy as np
import pandas as pd
import gadgets as gd

data_config = None
with open("read_data_config.yml", "r") as config_file:
    data_config = yaml.safe_load(config_file)

def query_formatter(**kwargs):
    return data_config["sme_query_template"].format(**kwargs)

def connect_db(db_dict):
    return pymysql.connect(**db_dict)

def make_date_list(start_date: str, end_date: str, format='%Y-%m-%d'):
    start_date = dt.datetime.strptime(start_date, format).strftime(
        '%Y-%m-%d'
    )
    end_date = dt.datetime.strptime(end_date, format).strftime(
        '%Y-%m-%d'
    )
    start_end = (dt.datetime.strptime(end_date, format)
                 - dt.timedelta(days=1)).strftime('%Y-%m-%d')
    end_start = (dt.datetime.strptime(start_date, format)
                 + dt.timedelta(days=1)).strftime('%Y-%m-%d')

    date_range = np.concatenate(
        [np.arange(start_date,
                   start_end,
                   dtype='datetime64[D]').reshape(-1, 1),
         np.arange(end_start,
                   end_date,
                   dtype='datetime64[D]').reshape(-1, 1)],
        axis=1)

    return date_range

def make_date_list_from_datetime(start_date, end_date):
    return make_date_list(str(start_date), str(end_date + dt.timedelta(2)))


def make_data_dict(start_date, end_date):
    sql_list_str = lambda l: ",\n".join(l)
    list_str = lambda l: '"' + '",\n"'.join(l) + '"'
    data_dict = {
        **data_config,
        "date_start": start_date,
        "date_end": end_date,
        "feature_list_str": sql_list_str(data_config['raw_feature_list']),
        "eng_feature_list_str": sql_list_str(data_config['eng_feature_list'])
    }
    data_dict['sme_query'] = query_formatter(**data_dict)
    data_dict['date_list'] = make_date_list_from_datetime(start_date, end_date)
    data_dict['db_connector'] = lambda **kwargs: connect_db(data_dict['db_args'])
    return data_dict

def read_data(data_dict):
    production_reader = gd.sql.MultiSQL(connector=data_dict['db_connector'])
    print(">>> start: {}".format(time.asctime()))
    data = gd.sql.trans.lower_columns(
    production_reader.get_data(
        query_gen=gd.sql.make_query_gen(
            data_dict['sme_query'],
            vals=data_dict['date_list']),
        threads = data_dict['read_threads']
    ))
    print("<<< finished sme read: {}".format(time.asctime()))
    return data