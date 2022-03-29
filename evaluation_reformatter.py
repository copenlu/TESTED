import os
import uuid
import json
import argparse
from tqdm import tqdm
from utils import write_jsonl
from typing import List, Dict, NoReturn

def data_reader(data_path:str = 'English') -> Dict:
    """Data Reader function

    Args:
        data_path (str): Path to the original data folder. Defaults to 'English'.

    Returns:
        Dict: Processed data Dictionary.
    """    
    data_complete = {}
    fundamental_keys = ['label', 'hypothesis','hypothesis']

    try:
        for data_folder in tqdm(os.listdir(data_path)):
            print(f"Unpacking {data_folder}")
            
            data_complete[data_folder] =[]
            specific_data_folder = f'{data_path}/{data_folder}'

            for data_file in tqdm(os.listdir(specific_data_folder)):
                
                data_abs_path = f'{specific_data_folder}/{data_file}'
                
                with open(data_abs_path) as file:
                    lines = file.readlines()
                    for line in lines:
                        data_line = json.loads(line)
                        
                        if not all(name in list(data_line.keys()) for name in fundamental_keys):
                            break
                        
                        data_complete[data_folder].append(data_line)
                print(data_file)
            
            if len(data_complete[data_folder]) < 1:
                print("Data Folder ")
                data_complete.pop(data_folder, None)

    except RuntimeError as e:
        print("Unable to read the Data Folder with error: ",e)
        return data_complete

    return data_complete

def check_lengths(data_complete:Dict = {}) -> NoReturn:
    data_lens = [(len(data_complete[i]),i) for i in data_complete]
    print(data_lens)


def reformat_data(unprocessed_path:str = '', save_path:str = "Data/evaluation_datasets") -> bool:
    """Reformat the data and write a JSONL file for each dataset.

    Args:
        unprocessed_path (str): Path to the original dataset. Defaults to ''.
        save_path (str): Path to save the processed dataset. Defaults to "Data/evaluation_datasets".

    Returns:
       bool : Successful or not.
    """
    success = False
    ending_list = ['.', '?', '!', '...']

    try:

        data_complete = data_reader(unprocessed_path)
        for data_key in data_complete:
            dedup_matrx = {}
            counter = 0

            data_processed = []
            labels = []


            for data_meta in tqdm(data_complete[data_key]):
                
                text_id_unique = uuid.uuid4().int
                user_id_unique = uuid.uuid4().int
                
                premise = data_meta['premise']
                hypothesis = data_meta['hypothesis']            
                    
                data_meta_updated = {"id": text_id_unique, "id_str": str(text_id_unique), \
                "premise": premise, "hypothesis":hypothesis , "lang":'en', 'usr': {'id': user_id_unique, "id_str": str(user_id_unique)},\
                "label": data_meta['label']}
                
                labels.append(data_meta['label'])
                
                if hash(premise+hypothesis) not in dedup_matrx:
                    dedup_matrx[hash(premise+hypothesis)] = 1
                else:
                    counter+= 1
                    continue
                    
                data_processed.append(data_meta_updated)
            
            success = write_jsonl(data = data_processed, data_path=save_path,data_key=data_key)
            assert success > 0
    
    except RuntimeError as e:
        print("Unable to Process and write the ")
        return success
    return success


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Reformat the data for evaluation')
    parser.add_argument('--unprocessed_path', type=str, default='English', help='Path to the unprocessed data')
    parser.add_argument('--save_path', type=str, default='Data/evaluation_datasets', help='Path to save the reformatted data')
    args = parser.parse_args()
    
    
    reformat_data(unprocessed_path=args.unprocessed_path, save_path=args.save_path)
