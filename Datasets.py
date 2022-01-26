import torch
import numpy as np
from typing import Dict
import pickle, hgtk, os

from Pattern_Generator import Pattern_Generate, Text_Filtering, Decompose

def Text_to_Token(text, token_dict):
    return np.array([
        token_dict[letter]
        for letter in ['<S>'] + list(text) + ['<E>']
        ], dtype= np.int32)

def Token_Stack(tokens, token_dict):
    max_token_length = max([token.shape[0] for token in tokens])
    tokens = np.stack(
        [np.pad(token, [0, max_token_length - token.shape[0]], constant_values= token_dict['<E>']) for token in tokens],
        axis= 0
        )
    return tokens

def Feature_Stack(features):
    max_feature_length = max([feature.shape[0] for feature in features])
    features = np.stack(
        [np.pad(feature, [[0, max_feature_length - feature.shape[0]], [0, 0]], constant_values= -10.0) for feature in features],
        axis= 0
        )
    return features

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        language_dict: Dict[str, int],
        pattern_path: str,
        metadata_file: str,
        feature_type: str,
        feature_length_min: int,
        feature_length_max: int,
        text_length_min: int,
        text_length_max: int,
        accumulated_dataset_epoch: int= 1,
        augmentation_ratio: float= 0.0
        ):
        super(Dataset, self).__init__()
        self.token_dict = token_dict
        self.language_dict = language_dict    
        self.pattern_path = pattern_path
        self.feature_type = feature_type

        if feature_type == 'Mel':
            feature_length_dict = 'Mel_Length_Dict'
        elif feature_type == 'Spectrogram':
            feature_length_dict = 'Spectrogram_Length_Dict'

        metadata_dict = pickle.load(open(
            os.path.join(pattern_path, metadata_file).replace('\\', '/'), 'rb'
            ))
        
        self.patterns = []
        max_pattern_by_speaker = max([len(patterns) for patterns in metadata_dict['File_List_by_Speaker_Dict'].values()])
        for patterns in metadata_dict['File_List_by_Speaker_Dict'].values():
            ratio = float(len(patterns)) / float(max_pattern_by_speaker)
            if ratio < augmentation_ratio:
                patterns *= int(np.ceil(augmentation_ratio / ratio))
            self.patterns.extend(patterns)
        
        self.patterns = [
            x for x in self.patterns
            if all([
                metadata_dict[feature_length_dict][x] >= feature_length_min,
                metadata_dict[feature_length_dict][x] <= feature_length_max,
                metadata_dict['Text_Length_Dict'][x] >= text_length_min,
                metadata_dict['Text_Length_Dict'][x] <= text_length_max
                ])
            ] * accumulated_dataset_epoch

    def __getitem__(self, idx):
        path = os.path.join(self.pattern_path, self.patterns[idx]).replace('\\', '/')
        pattern_dict = pickle.load(open(path, 'rb'))
        
        feature = pattern_dict[self.feature_type]
        token = Text_to_Token(pattern_dict['Decomposed'], self.token_dict)

        language = self.language_dict[pattern_dict['Language']]
        
        return feature, token, language

    def __len__(self):
        return len(self.patterns)

class Collater:
    def __init__(
        self,
        token_dict: Dict[str, int]
        ):
        self.token_dict = token_dict

    def __call__(self, batch):
        features, tokens, languages = zip(*batch)
        feature_lengths = [feature.shape[0] for feature in features]
        token_lengths = [token.shape[0] for token in tokens]        

        features = Feature_Stack(features)
        tokens = Token_Stack(tokens, self.token_dict)
        
        features = torch.FloatTensor(features).permute(0, 2, 1)   # [Batch, Feature_dim, Time]
        feature_lengths = torch.LongTensor(feature_lengths)   # [Batch]
        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        token_lengths = torch.LongTensor(token_lengths)   # [Batch]
        languages = torch.LongTensor(languages) # [Batch]

        return features, feature_lengths, tokens, token_lengths, languages


class Token_Interpreter:
    def __init__(self, token_dict):
        self.token_dict = {
            index: token
            for token, index in token_dict.items()
            }
        self.token_dict[len(token_dict)] = '<B>'

    def Interpreter(self, tokens):
        return [self.token_dict[token] for token in tokens]

    def Deduplication(self, tokens):
        new_tokens = [tokens[0]]
        for token in tokens[1:]:
            if token != new_tokens[-1]:
                new_tokens.append(token)

        return new_tokens

    def Clean(self, tokens):
        tokens = [
            x for x in tokens
            if not x in ['<S>', '<E>', '<B>']
            ]
        
        return tokens

    def Kor_Compose(self, tokens):
        return hgtk.text.compose(''.join(tokens).replace('_', hgtk.text.DEFAULT_COMPOSE_CODE))