import numpy as np 
import pandas as pd 
import os 
import re 
import nltk 
import string 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
import logging 
 
#logging
logger = logging.getLogger('data_preprocessing') 
logger.setLevel(logging.DEBUG) 

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('preprocessing_errors.log') 
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 

console_handler.setFormatter(formatter) 
file_handler.setFormatter(formatter) 

logger.addHandler(console_handler) 
logger.addHandler(file_handler)  

nltk.download('stopwords') 
nltk.download('wordnet') 


def preprocess_comment(comment): 
    """ Apply preprocessing to a comment"""   
    try:  
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment) 

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment

    except Exception as e:
        logger.error(f"Error preprocessing comment: {e}")
        return comment

def normalize_text(df): 
    """ Normalize the text DataFrame """   
    try:  
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment) 
        logger.debug('Text normalization completed.') 
        return df
    except Exception as e: 
        logger.error(f"Error normalizing text: {e}") 
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str)-> None:
    """ Save the train and test data"""  
    try:  
        interim_data_path = os.path.join(data_path, 'interim') 
        logger.debug(f'Creating Directory {interim_data_path}')  

        os.makedirs(interim_data_path, exist_ok=True) 
        logger.debug(f'Directory {interim_data_path} created successfully or already exists.') 

        train_data.to_csv(os.path.join(interim_data_path, 'train_processed.csv'), index=False) 
        test_data.to_csv(os.path.join(interim_data_path, 'test_processed.csv'), index=False)

        logger.debug(f'Data saved successfully to {interim_data_path}')
    except Exception as e: 
        logger.error(f"Error saving data: {e}") 
        raise

def main(): 
    try: 
        logger.debug('Starting data preprocessing...') 

        #fetch data 
        train_data = pd.read_csv('data/raw/train.csv')  
        test_data = pd.read_csv('data/raw/test.csv')   

        logger.debug('Data fetched successfully.')  

        train_processed_data = normalize_text(train_data) 
        test_processed_data = normalize_text(test_data)

        save_data(train_processed_data, test_processed_data, './data') 
        logger.debug('Data preprocessing completed successfully.')
    except Exception as e: 
        logger.error(f"Error in main preprocessing function: {e}") 
        raise

      
if __name__ == "__main__": 
    main()