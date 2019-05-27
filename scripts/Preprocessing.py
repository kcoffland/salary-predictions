import pandas as pd

class Preprocessing:
    """
    Processes the data and ensures that the test data cannot learn
    more information that what the training data will give (e.g. new
    columns will be filled with 0s to give no additional information).
    """

    def __init__(self, cols_to_filter=None):
        """
        Args:
            cols_to_filter: list of strings, names columns which need
                            to be removed from the dataset
        """

        # stores whether the model processor has been fit to the data
        self.is_fit = False

        assert(type(cols_to_filter) == list)
        self.cols_to_filter = cols_to_filter

    def fit(self, X, y=None):
        """
        Finds out what which columns need to be converted to dummy
        values and stores it as an attribute to be used later in
        the transform method.
        """
        self.is_fit = True

        # Defined here instead of __init__ because the constructor
        # won't know the data being passed in, and I don't want to
        # make unnecessary assignments. I've ensured in the rest of
        # the methods that fit must be called before the rest
        self.categorical_cols = pd.Index([x for x in X.columns
                                            if X[x].dtype == object \
                                                and x not in self.cols_to_filter
                                          ]
                                   )

        return self

    def transform(self, X, y=None):

        # Ensuring data has been fit before this method can be used
        if not self.is_fit:
            raise Error("Fit method must be called before data can be transformed")

        # Removing columns that the user wants to filter
        for col in self.cols_to_filter:
            assert(col in X.columns and "Given filter column not in data")
        X.drop(self.cols_to_filter, inplace=True, axis=1)

        # Setting all values for columns not in the training data to 0
        # so no additional information can be learned
        new_cat_cols = pd.Index([x for x in X.columns
                                  if X[x].dtype == object])

        new_cols = set(new_cat_cols) - set(self.categorical_cols)

        for col in new_cols:
            X[col] = 0

        X_new = pd.get_dummies(X)

        return X_new

    def fit_transform(self, X, y=None):

        return self.fit(X).transform(X)