import torch
import pymongo



class WeakLabelerSingelton:
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if WeakLabelerSingelton.__instance == None:
            WeakLabelerSingelton()
        return WeakLabelerSingelton.__instance


    def set_attr(self, **kwargs):

        print(kwargs.keys())
        if 'model' in kwargs:
            self.model =  kwargs['model']
        if 'tokenizer' in kwargs:
            self.tokenizer = kwargs['tokenizer']

        if 'pipeline' in kwargs:
            self.pipeline = kwargs['pipeline']

        if 'labeler_metadata' in kwargs:
            self.labeler_metadata = kwargs['labeler_metadata']

        if 'mongo_client' in kwargs:
            self.mongo_client = pymongo.MongoClient(kwargs['mongo_client'])
        if 'mongodb' in kwargs:
            self.mongodb = self.mongo_client[kwargs['mongodb']]
        if 'collection' in kwargs:
            self.collection = self.mongodb[kwargs['collection']]
        if 'data_loader' in kwargs:
            self.data_loader = kwargs['data_loader']
        if 'insertion_scheme' in kwargs:
            self.insertion_scheme = kwargs['insertion_scheme']

        self.cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__instance = self

    def __init__(self, model = None, tokenizer = None, pipeline = None, labeler_metadata:dict = {}, mongo_server:str  =  "mongodb://localhost:27017/" ):
        """ Virtually private constructor. """
        if WeakLabelerSingelton.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            WeakLabelerSingelton.model = model
            WeakLabelerSingelton.tokenizer = tokenizer
            WeakLabelerSingelton.pipeline = pipeline
            WeakLabelerSingelton.labeler_metadata = labeler_metadata

            WeakLabelerSingelton.cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            WeakLabelerSingelton.__instance = self

            WeakLabelerSingelton.mongo_client = pymongo.MongoClient(mongo_server)
            WeakLabelerSingelton.mongodb = WeakLabelerSingelton.mongo_client["Users"]
            WeakLabelerSingelton.collection = WeakLabelerSingelton.mongodb["Processed_Tweets"]
