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

def find_outliers(data):
    """Finds the values for which outliers begin based on 1.5*iqr
    Args:
        data: Series, holds data you want to find the outliers for
    Returns:
        lower, upper: floats, values less than lower are outliers and
                              values larger than upper are outliers
    """

    # Finding the interquartile range
    q1 = data.quantile(.25)
    q3 = data.quantile(.75)
    iqr = q3-q1

    upper = q3 + iqr*1.5
    lower = q1 - iqr*1.5

    return lower, upper

def attended_college(df):
    """Tests the given dataframe to see if the instance requires college
        Function is specific to the salary-predictions project
    Args:
        df: DataFrame, holds features that the function will be applied to
    Returns:
        bool, True if the feature needs college and False if it doesn't
    """

    if df.degree == 'NONE' or df.degree == 'HIGH_SCHOOL':
        return False
    else:
        return True

def grad_types(df):
    """Determines the grad type of the instance
        Function is specific to salary-predictions project
        This must be run after attended_college
    Args:
        df: DataFrame, holds the instances we want to test
    Returns:
        string, Denotes Education and experience level
    """

    assert('college' in df.columns)
    assert('ExperienceLevel' in df.columns)

    if df['college']:
        if df['ExperienceLevel'] == 0:
            return 'No College, Inexperienced'
        elif df['ExperienceLevel'] == 1:
            return 'No College, Some Real Experience'
        else:
            return 'No College, Real World Experienced'
    else:
        if df['ExperienceLevel'] == 0:
            return 'Recent Grad'
        elif df['ExperienceLevel'] == 1:
            return 'Some Experience Grad'
        else:
            return 'Experienced Grad'
        
        
        
        