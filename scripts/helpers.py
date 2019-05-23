import pandas as pd
import os

def load_data(path, file, verbose=False, index=0):
    """Reads in the data at the given path/filename. Currently file is assumed to be csv 
    Args:
        path: string, path to file containing data
        file: string, name of file containing data that will be merged together
        verbose: boolean, If the caller wanted information about the df to be displayed
    Returns: 
        df, holds data in the given csv files
    """
    
    df = pd.read_csv(path+file, index_col=index)
    
    if verbose:
        shape = f'{df.shape}'
        dtypes = f'{df.dtypes[:30]}'
        head = f'{df.head()[:10]}'
        name = file.split('.')[0]
        
        print(f'{name} shape'.center(80, '-'))
        print(shape.center(80))
        print(f"{name}'s column types".center(80, '-'))
        print(dtypes)
        print(f"{name} first five rows".center(80, '-'))
        print(head)
    
    return df

def mean_squared_error(predictions, actual):
    """Computes the MSE for a given prediction and observed data
    Args:
         predictions: Series or float/int, holding predicted target value
         actual: Series, holding actual target value
    """

    return ((predictions-actual)**2).sum() / len(actual)

        
        
        
        
        
        