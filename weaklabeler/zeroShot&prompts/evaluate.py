import os
import json
import torch
import pymongo
import argparse
from tqdm import tqdm
from weak_labeler.tools.weaklabeler.tools.utils import WeakLabelerSingelton
from typing import List, Dict, NoReturn



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate the results of Weak Labeler on a task') 

    parser.add_argument(
    '--mongo_db',type=str,default='Users',
    help="The name of the MongoDB database"
    )

    parser.add_argument(
    '--mongo_collection',type=str,default='Processed_Tweets',
    help="The name of the MongoDB collection"
    )

    parser.add_argument(
    '--mongo_client',type=str,default='mongodb://localhost:27017/',
    help="The name of the MongoDB client (ip:port)"
    )

    parser.add_argument(
    '--original_verbalizers',type=str,default=True,dest='original_verbalizers',
    help="The location of original verbalizers"
    )

    parser.add_argument(
    '--new_verbalizers',type=str,default=True,dest='new_verbalizers',
    help="The location of verbalizers used in the weak labeler"
    )


    args = parser.parse_args()

    mongo_client, mongo_db, mongo_collection  = args.mongo_client, args.mongo_db, args.mongo_collection


    env = WeakLabelerSingelton.getInstance()


        