import json
from weaklabeler.tools.utils import insert_scheme
from env_loader import WeakLabelerSingelton

class DataLoader:
    def __init__(self, data_path: str = '') -> None:
        
        self.data_path = data_path
    
    def read_data(self, dataset_path: str = '', batch_size = 1):
        user_data = []
        try:

            batch = []
            with open(dataset_path) as file:
                for line in file:
                    data = json.loads(line)
                    batch.append(data)
                    if len(batch) == batch_size:
                        yield batch
                        batch = []

                if len(batch) > 0:
                    yield batch
                    batch = []

        except RuntimeError as e:
            print('Cannot read the user data with error: ',e)

    def data_generator(self):
        language_groups = {}
        env = WeakLabelerSingelton.getInstance()

        for data_lines in self.read_data(self.data_path):
            for data_line in data_lines:
                if data_line['lang'] not in language_groups:
                    language_groups[data_line['lang']] = []

                if 'full_text' in data_line:
                    language_groups[data_line['lang']].append({'full_text': data_line['full_text'], \
                                                        'tweet_id':data_line['id'],'lang':data_line['lang'],\
                                                        'user_id':data_line['user']['id'] })
                else:
                    full_text = insert_scheme(data_line['premise'], data_line['hypothesis'],env.insertion_scheme)

                    language_groups[data_line['lang']].append({'full_text': full_text, \
                                                        'tweet_id':data_line['id'],'lang':data_line['lang'],\
                                                        'user_id':data_line['user']['id'],'label':data_line['label']})

                if len(language_groups[data_line['lang']]) >= 16:
                    yield language_groups[data_line['lang']]
                    language_groups[data_line['lang']] = []

        for lang in language_groups:
            if len(language_groups[lang]) != 0:
                yield language_groups[lang]

