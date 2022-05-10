from tqdm import tqdm
from typing import List, Dict, Union, NoReturn
import numpy as np


def load_pretrained_vectors(word2index: Dict = {}, file_path: str = '')-> np.array:
    """Load pretrained vectors and create embedding layers.
    
    Args:
        word2index (Dict): Vocabulary built from the corpus
        file_path (str): Path to embedding file

    Returns:
        embeddings (np.array): Embedding matrix (n_words, dim) 
    """

    print("Loading pretrained vectors...")

    with open(file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as file:
        n_words, dim = map(int, file.readline().split())

        # Initilize random embeddings
        embeddings = np.random.uniform(-0.25, 0.25, (len(word2index), dim))
        embeddings[word2index['<pad>']] = np.zeros((dim,))

        # Load pretrained vectors
        count = 0
        for line in tqdm(file):
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            if word in word2index:
                count += 1
                embeddings[word2index[word]] = np.array(tokens[1:], dtype=np.float32)

        print(f"There are {count} / {len(word2index)} pretrained vectors found.")

    return embeddings