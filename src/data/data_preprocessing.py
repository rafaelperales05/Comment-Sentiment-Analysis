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

def normalize_text(df): 
    """ Normalize the text DataFrame """  

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str)-> None:
    """ Save the train and test data""" 

def main():  
    pass
     
if __name__ == "__main__": 
    main()