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


    def set_attr(self, model = None, tokenizer = None, pipeline = None, labeler_metadata = None, mongo_client = None, mongodb = None, collection = None ):
        if model is not None:
            self.model = model
        if tokenizer is not None:
            self.tokenizer = tokenizer

        if pipeline is not None:
            self.pipeline = pipeline

        if labeler_metadata is not None:
            self.labeler_metadata = labeler_metadata

        if mongo_client is not None:
            self.mongo_client = pymongo.MongoClient(mongo_client)
        if mongodb is not None:
            self.mongodb = self.mongo_client[mongodb]
        if collection is not None:
            self.collection = self.mongodb[collection]

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
