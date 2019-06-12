import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from helpers import attended_college, grad_types, tech_mogul, oil_baron

class Preprocessing:
    """
    Processes the data and ensures that the test data cannot learn
    more information that what the training data will give (e.g. new
    columns will be filled with 0s to give no additional information).
    """

    def __init__(self, cols_to_filter=None, ordinal_cols=None, binned_cols=None, inplace=False,
                 grad_types=False, tech_mogul=False, oil_baron=False, oil_rig=False, combo=False):
        """
        Args:
            cols_to_filter: list of strings, names columns which need
                            to be removed from the dataset
            ordinal_cols: dict {string: list of strings}, the key is the ordinal
                            column and the value is the list of categories in
                            order of importance. ORDER OF IMPORTANCE WILL MATTER
            binned_cols: dict {string: int}, the key is the column which
                            the user wants to be binned. The int will be the number of
                            bins that will be created.The data will be binned into equal
                            quantiles using pd.qcut(). The new column will be added to
                            the dataframe as 'given_binned' where "given" is the key of
                            the dict. If the user wants the original column removed, then
                            they can add it to the columns to filter. Note: some engineered
                            features require the binning of certain columns and will
                            not work if they have not been binned.
            grad_types: bool, if True, then the grad_types feature will be created.
                            ONLY TO BE USED FOR SALARY-PREDICTIONS
        """

        # stores whether the model processor has been fit to the data
        self.is_fit = False

        if cols_to_filter:
            assert(type(cols_to_filter) == list)

        self.cols_to_filter = cols_to_filter

        self.categorical_cols = []
        self.numerical_cols = []
        self.dummy_cols = []

        # Setting up ordinal columns encoding
        self.ordinal_cols = ordinal_cols
        if self.ordinal_cols:
            assert(type(ordinal_cols) == dict)
            categories = list(ordinal_cols.values())
            self.ordinal_enc = OrdinalEncoder(categories=categories)
        else:
            self.ordinal_enc = None

        # scaler to make numerical columns between 0 and 1
        self.min_max_scaler = MinMaxScaler()

        # Setting up binned columns
        self.binned_cols = binned_cols
        if self.binned_cols:
            self.bins = [[] for i in range(len(self.binned_cols))]

        # Giving the user the option to transform data inplace
        self.inplace = inplace

        # Adding optional features
        self.grad_types = grad_types
        self.tech_mogul = tech_mogul
        self.oil_baron = oil_baron
        self.oil_rig = oil_rig
        self.combo = combo


    def fit(self, X, y=None):
        """
        Finds out what which columns need to be converted to dummy
        values and stores it as an attribute to be used later in
        the transform method.
        """
        self.is_fit = True

        # Creating the bins from the training data to cut on later
        if self.binned_cols:
            for i, col in enumerate(self.binned_cols.keys()):
                assert (col in X.columns)
                column, self.bins[i] = pd.qcut(X[col], self.binned_cols[col],
                                            labels=False, retbins=True)


        self.categorical_cols = pd.Index([x for x in X.columns
                                          if X[x].dtype == object
                                          ]
                                         )

        self.numerical_cols = pd.Index([x for x in X.columns
                                        if x not in self.categorical_cols
                                        ]
                                       )

        # Removing filtered columns from established numeric and categorical columns
        if self.cols_to_filter:
            for col in self.cols_to_filter:
                if col in self.categorical_cols:
                    self.categorical_cols = self.categorical_cols.drop([col])
                elif col in self.numerical_cols:
                    self.numerical_cols = self.numerical_cols.drop([col])

        # Finding all columns that are dummied, note this is added after columns
        # are filtered out of self.categorical_cols
        if not self.categorical_cols.empty:
            # Ensuring ordinal cols aren't dummied
            if self.ordinal_cols:
                nominal = set(self.categorical_cols) - set(self.ordinal_cols.keys())
            else:
                nominal = set(self.categorical_cols)
            self.dummy_cols = pd.get_dummies(X.loc[:, nominal]).columns

        # Creating ordinal columns
        if self.ordinal_cols:
            for col in self.ordinal_cols.keys():
                assert(col in X.columns)
            self.ordinal_enc.fit(X[list(self.ordinal_cols.keys())])

        # Fitting the scaler to the training data's numerical columns
        if not self.numerical_cols.empty:
            self.min_max_scaler.fit(X[self.numerical_cols])

        return self

    def transform(self, X, y=None):
        # Ensuring data has been fit before this method can be used
        if not self.is_fit:
            raise RuntimeError("Fit method must be called before data can be transformed")

        # maybe this naming convention is why software engineers hate Data Scientists
        if self.inplace:
            X_new = X
        else:
            X_new = X.copy()

        # Setting all values for columns not in the training data to 0
        # so no additional information can be learned outside of engineering
        # new features available in the training data
        if self.binned_cols:
            new_cols = set(X_new.columns) - set(self.categorical_cols) - \
                       set(self.numerical_cols) - set(self.binned_cols.keys())
        else:
            new_cols = set(X_new.columns) - set(self.categorical_cols) - \
                       set(self.numerical_cols)
        for col in new_cols:
            X_new[col] = 0

        ################ Creating/Removing Columns ##################
        # Binning appropriate columns
        if self.binned_cols:
            for i, col in enumerate(self.binned_cols.keys()):
                # I cast to int so that I don't have to worry about the Category dtype
                # If I don't include_lowest, then job postings with 0 experience
                # will not be included in the bin
                X_new[f'{col}_binned'] = pd.cut(X_new[col], self.bins[i],
                                                labels=False, include_lowest=True)\
                                                .astype(int).copy()
                X_new[f'{col}_binned'] /= X_new[f'{col}_binned'].max()

        # Adding any new features
        if self.grad_types:
            assert('yearsExperience' in self.binned_cols.keys())
            X_new['attendedCollege'] = X_new.apply(attended_college, axis=1)
            X_new['gradTypes'] = X_new.apply(grad_types, axis=1)
            # Scaling it here because it won't be in the numerical_cols
            X_new['gradTypes'] /= X_new['gradTypes'].max()
        if self.tech_mogul:
            X_new['techMogul'] = X_new.apply(tech_mogul, axis=1)
        if self.oil_baron:
            X_new['oilBaron'] = X_new.apply(oil_baron, axis=1)

        # Removing columns that the user wants to filter
        if self.cols_to_filter:
            for col in self.cols_to_filter:
                assert(col in X_new.columns and f"Given column {col} not in data")

            X_new.drop(self.cols_to_filter, inplace=True, axis=1)

        ################### Transforming Data ########################
        # Encoding Ordinal columns as well as ensuring they're scaled properly
        if self.ordinal_cols:
            X_new[list(self.ordinal_cols.keys())] = self.ordinal_enc.transform(
                                                        X_new[list(self.ordinal_cols.keys())]
                                                    )
            X_new[list(self.ordinal_cols.keys())] /= X_new[list(self.ordinal_cols.keys())].max()

        # Dummy the rest of the categorical variables
        X_new = pd.get_dummies(X_new)
        if not self.dummy_cols.empty:
            for col in self.dummy_cols:
                if col not in X_new.columns:
                    X_new[col] = 0

        # Scale the numerical columns
        if not self.numerical_cols.empty:
            X_new[self.numerical_cols] = self.min_max_scaler.transform(X_new[self.numerical_cols])

        # This feature is added here because it works much better once the other features are scaled
        if self.combo:
            X_new['combo'] = X_new['yearsExperience'] / (X_new['milesFromMetropolis']+1)
            if 'yearsExperience' in self.binned_cols.keys():
                X_new.drop('yearsExperience_binned', axis=1, inplace=True)

        return X_new

    def fit_transform(self, X, y=None):

        return self.fit(X).transform(X)


