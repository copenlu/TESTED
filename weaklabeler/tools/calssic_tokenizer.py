from nltk.tokenize import word_tokenize
from collections import defaultdict

from typing import List, Dict, Union, NoReturn
import numpy as np


def tokenize(texts: List[str] = None) -> Union[List[str], Dict, int]:
    """
    This is a pipeline for "classic" text tokenization
    
    Args:
        texts (List[str]): A List of the texts for tokenization
    
    Returns:
        tokenized_texts (List[List[str]]): Tokens (List of Lists)
        word2index (Dict): Vocabulary from the data
        max_len (int): Maximum sentence length
    """

    max_len = 0
    tokenized_texts = []
    word2index = {}

    # Classical embedding usually lack padding and unknown tokens, Lets add them
    word2index['<pad>'] = 0
    word2index['<unk>'] = 1

    # Building our vocab from the data. Shifting for <pad> and <unk> => index = 2
    index = 2
    for text in texts:
        tokenized_text = word_tokenize(text)

        # Add `tokenized_text` to `tokenized_texts`
        tokenized_texts.append(tokenized_text)

        # Add new token to `word2idx`
        for token in tokenized_text:
            if token not in word2index:
                word2index[token] = index
                index += 1

        max_len = max(max_len, len(tokenized_text))

    return tokenized_texts, word2index, max_len

def encode(tokenized_texts: List[List[str]] = None, word2index: Dict = None, max_len: int = 0) -> np.array:
    """
    This is a pipeline for encoding the tokens into ID's adn padding to the
    maximum length of the vocabulary

    Returns:
        input_ids (np.array): Array of token indexes in the vocabulary with
            shape (N, max_len).
    """

    input_ids = []
    for tokenized_sent in tokenized_texts:
        # Padding to max_len
        tokenized_sent += ['<pad>'] * (max_len - len(tokenized_sent))

        # Encode 
        input_id = [word2index.get(token) for token in tokenized_sent]
        input_ids.append(input_id)
    
    return np.array(input_ids)