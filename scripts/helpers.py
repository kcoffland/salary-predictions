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

def find_outliers(data, method='iqr'):
    """Finds the values for which outliers begin
    Args:
        data: Series, holds data you want to find the outliers for
        method: string, Method used to calculate an outlier
    Returns:
        lower, upper: floats, values less than lower are outliers and
                              values larger than upper are outliers
    """

    if method=='iqr':
        # Finding the interquartile range
        q1 = data.quantile(.25)
        q3 = data.quantile(.75)
        iqr = q3-q1

        upper = q3 + iqr*1.5
        lower = q1 - iqr*1.5
    elif method=='std':
        std = data.std()
        lower = data.mean() - 3*std
        upper = data.mean() + 3*std
    else:
        raise ValueError("Invalid value for 'method' passed")


    return lower, upper

def attended_college(df):
    """Tests the given dataframe to see if the instance requires college
        Function is specific to the salary-predictions project
    Args:
        df: DataFrame, holds features that the function will be applied to
    Returns:
        bool, True if the feature needs college and False if it doesn't
    """

    # Checking to see if the posting requires a college degree
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

    # Ensures df has the proper columns
    assert('attendedCollege' in df.index)
    assert('yearsExperience_binned' in df.index)

    # Gives a ranking based on combination of education and experience
    if df['attendedCollege']:
        if df['yearsExperience_binned'] == 0:
            return 3
        elif df['yearsExperience_binned'] == .25:
            return 5
        elif df['yearsExperience_binned'] == .5:
            return 7
        elif df['yearsExperience_binned'] == .75:
            return 8
        else:
            return 9
    else:
        if df['yearsExperience_binned'] == 0:
            return 0
        elif df['yearsExperience_binned'] == .25:
            return 1
        elif df['yearsExperience_binned'] == .5:
            return 2
        elif df['yearsExperience_binned'] == .75:
            return 4
        else:
            return 6

def tech_mogul(df):
    assert('jobType' in df.index and 'jobType column does not exist')
    assert('industry' in df.index and 'industry column does not exist')

    moguls = ['CEO', 'CTO', 'CFO']

    if df['industry']== 'WEB' and df['jobType'] in moguls:
        return 1
    else:
        return 0

def oil_baron(df):
    assert('jobType' in df.index and 'jobType column does not exist')
    assert ('industry' in df.index and 'industry column does not exist')

    barons = ['CEO', 'CTO', 'CFO']

    if df['industry'] == 'OIL' and df['jobType'] in barons:
        return 1
    else:
        return 0

        