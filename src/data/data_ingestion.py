import numpy as np 
import pandas as pd 
import os 
import logging 
import yaml 
from sklearn.model_selection import train_test_split 
#logging
logger = logging.getLogger('data_ingestion') 
logger.setLevel(logging.DEBUG) 

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log') 
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 

console_handler.setFormatter(formatter) 
file_handler.setFormatter(formatter) 

logger.addHandler(console_handler) 
logger.addHandler(file_handler)

def load_param(params_path : str) -> dict: 
    """ Load params from YAML""" 
    try: 
        with open(params_path, 'r') as file: 
            params = yaml.safe_load(file) 
        logger.debug('Parmaeters retrieved from %s', params_path) 
        return params 
    except FileNotFoundError: 
        logger.error('Parameters file not found: %s', params_path) 
        raise 
    except yaml.YAMLError as e: 
        logger.error('Error parsing YAML file: %s', e) 
        raise 
    except Exception as e: 
        logger.error('An unexpected error occurred: %s', e) 
        raise

def load_data(data_url: str) -> pd.DataFrame: 
    """Load Data from CSV file""" 
    try:  
        df = pd.read_csv(data_url) 
        logger.debug('Data loaded from %s', data_url) 
        return df 
    except pd.errors.ParserError as e: 
        logger.error('Error parsing CSV file: %s', e) 
        raise 
    except Exception as e: 
        logger.error('An unexpected error occurred while loading data: %s', e) 
        raise


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """ preprocess data by removing missing, dublicates and empty strings""" 
    try: 
        data.dropna(inplace=True)  
        data.drop_duplicates(inplace=True) 
        data = data[data['clean_comment'].str.strip() != '']  # Remove empty strings 

        logger.debug('Data preprocessing completed. Missing values, duplicates and empty strings are removed.' ) 
        return data 
    except KeyError as e:  
        logger.error('KeyError occurred during data preprocessing: %s', e) 
        raise
    except Exception as e: 
        logger.error('An unexpected error occurred during data preprocessing: %s', e) 
        raise 


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """ Save Train and test data, creating folder if it doesnt exist""" 
    try: 
        raw_data_path = os.path.join(data_path, 'raw') 
        os.makedirs(raw_data_path, exist_ok=True)  # Create directory if it doesn't exist 
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False) 
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False) 
        logger.debug('Data saved to %s', raw_data_path) 
    except Exception as e: 
        logger.error('An unexpected error occurred while saving data: %s', e) 
        raise

def main():  
    try:   
        # load 
        params = load_param(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
        test_size = params['data_ingestion']['test_size'] 
        df = load_data(data_url='https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')

        final_df = preprocess_data(data=df) 
        train_data , test_data = train_test_split(final_df, test_size=test_size, random_state=42) 
        save_data(train_data=train_data, test_data=test_data, data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data'))
    except Exception as e: 
        logger.error('An unexpected error occurred in the main function: %s', e)
        raise


if __name__ == '__main__': 
    main()