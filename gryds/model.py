from abc import ABC, abstractmethod


class GrydModel(ABC):

    @abstractmethod
    def fit(data, **kwargs):
        ''' Train the model with the given data '''
        pass

    @abstractmethod
    def set_parameters(**kwargs):
        ''' Set tunning parameters for this model '''
        pass

