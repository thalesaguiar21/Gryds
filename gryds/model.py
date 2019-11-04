class GrydModel(ABC):

    def fit(data, **kwargs):
        ''' Train the model with the given data '''
        pass

    def predict(data, **kwargs):
        ''' Predict the classes of the given samples '''
        pass

    def set_params(**kwargs):
        ''' Set tunning parameters for this model '''
        pass

