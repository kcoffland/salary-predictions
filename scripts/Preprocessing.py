

class Preprocessing:
    """
    Processes the data and ensures that the test data cannot learn
    more information that what the training data will give (e.g. new
    columns will be filled with 0s to give no additional information).
    """

    def __init__(self):
        self.is_fit = False

    def fit(self, X, y=None):
        """
        Finds out what which columns need to be converted to dummy
        values and stores it as an attribute to be used later in
        the transform method
        """
        self.is_fit = True

        self.categorical_cols = pd.Index([x for x in X.columns
                                            if X[x].dtype == object]
                                   )

        return self

    def transform(self, X, y=None):

        # Ensuring data has been fit before this method can be used
        if not self.is_fit:
            raise Error("Fit method must be called before data can be transformed")

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