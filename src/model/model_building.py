import numpy as np 
import pandas as pd 
import os 
import pickle 
import logging 
import yaml 
import lightgbm as lgb   
from sklearn.feature_extraction.text import TfidfVectorizer 
 

 #logging config 
logger = logging.getLogger('model_building') 
logger.setLevel(logging.DEBUG)    

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler) 



def load_param(params_path : str) -> dict: 
    """ load parameters from YAML file"""
    try:  
        with open(params_path, 'r') as file: 
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path) 
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

def load_data(file_path: str ) -> pd.DataFrame: 
    """ load data from csv""" 
    try:  
        df = pd.read_csv(file_path)  
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NANS filled from %s', file_path) 
        return df
    except pd.errors.ParserError as e: 
        logger.error('Error parsing data file: %s', e) 
        raise 
    except Exception as e: 
        logger.error('An unexpected error occurred while loading data: %s', e) 
        raise 

def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:  

    try:  
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range) 

        X_train = train_data['clean_comment'].values   
        y_train = train_data['category'].values 

        X_train_tfidf = vectorizer.fit_transform(X_train) 
        logger.debug(f"TF-IDF matrix shape: {X_train_tfidf.shape}") 
        with open(os.path.join(get_root_directory(), 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f) 
        
        logger.debug('TF-IDF transformation applied with max_features=%d and ngram_range=%s', max_features, ngram_range) 
        return X_train_tfidf, y_train 
    except Exception as e: 
        logger.error('An unexpected error occurred during TF-IDF transformation: %s', e) 
        raise 


def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int) -> lgb.LGBMClassifier: 
    """ Train LightGBM model"""  

    try:  
        best_model = lgb.LGBMClassifier( objective='multiclass', 
                                       num_class=3, 
                                       metric='multi_logloss', 
                                        is_unbalance=True, 
                                        class_weight='balanced', 
                                        reg_alpha=0.1, 
                                        reg_lambda=0.1,
                                        learning_rate=learning_rate,
                                       max_depth=max_depth, 
                                       n_estimators=n_estimators) 
        best_model.fit(X_train, y_train) 
        logger.debug('LightGBM model trained with learning_rate=%.4f, max_depth=%d, n_estimators=%d', learning_rate, max_depth, n_estimators) 
        return best_model
    except Exception as e: 
        logger.error('An unexpected error occurred during model training: %s', e) 
        raise

def save_model(model, file_path : str)-> None: 
    """ Save the trained model to a file"""
    try:  
        with open(file_path, 'wb') as f: 
            pickle.dump(model, f) 
        logger.debug('Model saved to %s', file_path)
    except Exception as e: 
        logger.error('An unexpected error occurred while saving the model: %s', e) 
        raise


def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../')) 


def main(): 
    try:
        root = get_root_directory()   

        params = load_param(os.path.join(root, 'params.yaml'))  

        max_features = params['model_building']['max_features'] 
        ngram_range = tuple(params['model_building']['ngram_range']) 
        learning_rate = params['model_building']['learning_rate'] 
        max_depth = params['model_building']['max_depth'] 
        n_estimators = params['model_building']['n_estimators'] 

        train_data = load_data(os.path.join(root, 'data/interim/train_processed.csv'))  
        logger.debug(f"Training data shape: {train_data.shape}")

        X_train_tfidf , y_train = apply_tfidf(train_data, max_features, ngram_range)    
        best_model = train_lgbm(X_train_tfidf, y_train, learning_rate, max_depth, n_estimators) 
        save_model(best_model, os.path.join(root, 'lgbm_model.pkl'))  
    except Exception as e: 
        logger.error('An error occurred in the main execution flow: %s', e) 
        raise 

if __name__ == '__main__':
    main()

     
