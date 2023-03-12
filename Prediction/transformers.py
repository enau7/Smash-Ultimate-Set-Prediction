class columnDropperTransformer:
    def __init__(self,columns):
        self.columns=columns

    def transform(self, X, y = None):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y = None):
        return self
    
class columnSelectorTransformer:
    def __init__(self,columns):
        self.columns=columns

    def transform(self, X, y = None):
        return X[self.columns]

    def fit(self, X, y = None):
        return self