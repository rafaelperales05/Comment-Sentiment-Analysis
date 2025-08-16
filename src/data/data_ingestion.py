import numpy as np 
import pandas as pd 
import os 
import logging 
import yaml 
from sklearn.model_selection import train_test_split 
 
 #logging 

def load_param(params_path : str) -> dict:
    pass  

def load_data(data_url: str) -> pd.DataFrame: 
    pass  

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # Implement your data preprocessing steps here
    pass 

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    pass

def main():  
    pass



if __name__ == '__main__': 
    main()