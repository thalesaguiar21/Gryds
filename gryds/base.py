class GrydModel:

    def fit(self, X, Y):
        ''' Train the model with the given data '''
        pass

    def predict(self, data):
        ''' Predict the classes of the given samples '''
        pass

    def set_params(self, **kwargs):
        ''' Set tunning parameters for this model '''
        pass


class Results:

    def __init__(self):
        self.scores = []
        self.traintimes = []
        self.testtimes = []

    def add(self, scr, train, test):
        self.scores.append(scr)
        self.traintimes.append(train)
        self.testtimes.append(test)


