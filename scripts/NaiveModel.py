class NaiveModel:
    """
    Creates a naive model given the column you want the data grouped
    by and the aggregate type you want to compare everything with. For
    example, if col is 'jobType' and agg_type is 'median', then the median
    value for the given job type would be returned as the predicted value
    """

    def __init__(self, agg_col, target_col, agg_type='mean'):
        """Constructor, sets that values for the current instance
        Args:
            agg_col: column that the aggregates will be grouped and performed on
            agg_type: string, aggregation method you want to use when creating
                      the naive model. Currently, only 'mean' and 'median' are
                      supported
        """
        # making sure inputs are correct
        assert ('mean' in agg_type.lower() or 'median' in agg_type.lower())

        self.agg_col = agg_col
        self.target_col = target_col
        self.agg_type = agg_type.lower()
        self.mapping = {}
        self.is_fit = False

    def fit(self, X, y=None):
        """Finds the aggregate value of the data split on the agg_col
        Args:
            X: Dataframe, holds all the data
            y: Only here for a future pipeline
        Returns:
            self: NaiveModel instance, for the fit->transform paradigm
        """

        assert (self.target_col in X.columns)
        assert (self.agg_col in X.columns)

        if self.agg_type == 'mean':
            self.mapping = X.groupby(self.agg_col)[self.target_col] \
                .mean() \
                .to_dict()
        elif self.agg_type == 'median':
            self.mapping = X.groupby(self.agg_col)[self.target_col] \
                .median() \
                .to_dict()

        self.is_fit = True

        return self

    def transform(self, X, y=None):
        """Does nothing since the data doesn't need to be cleaned at all
        Args:
            X: dataframe, holds all feature data
            y: series, target data
        Returns:
            X_new: dataframe, transformed feature data
        """

        if not self.is_fit:
            raise Error("Data must be fit before it can be transformed")
        X_new = X
        return X_new

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def predict(self, X):
        assert (self.agg_col in X)

        y_pred = X[self.agg_col].replace(self.mapping)
        return y_pred