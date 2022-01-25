import torch
import numpy as np
import logging, yaml, os, sys, argparse, time, math, pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Tuple
from librosa import griffinlim
from scipy.io import wavfile


from meldataset import spectral_de_normalize_torch

from Modules import Tacotron2
from Arg_Parser import Recursive_Parse

from Datasets import Text_to_Token, Inference_Collater
from Pattern_Generator import Text_Filtering, Decompose


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        ge2e_dict: Dict[str, int],
        texts: List[str],
        speakers: List[List[Tuple[str, float]]],
        emotion_ids: List[int]
        ):
        self.token_dict = token_dict
        self.ge2e_dict = ge2e_dict

        self.patterns = []
        for index, (text, speaker, emotion_id) in enumerate(zip(
            texts, speakers, emotion_ids
            )):
            text = Text_Filtering(text)
            if text is None or text == '':
                logging.warn('The text of index {} is incorrect. This line is ignoired.'.format(index))
                continue
            self.patterns.append((text, speaker, emotion_id))

    def __getitem__(self, idx):
        text, speakers, emotion_id = self.patterns[idx]
        
        decomposed_text = Decompose(text)

        ge2e = np.stack([
            self.ge2e_dict[speaker] * portion
            for speaker, portion in speakers
            ]).sum(axis= 0)
        
        return Text_to_Token(decomposed_text, self.token_dict), ge2e, emotion_id, text, decomposed_text, speakers

    def __len__(self):
        return len(self.patterns)

class Inferencer:
    def __init__(
        self,
        hp_path: str,
        checkpoint_path: str,
        out_path: str,
        batch_size= 1
        ):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        self.hp = Recursive_Parse(yaml.load(
            open(hp_path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        self.model = Tacotron2(self.hp).to(self.device)
        self.vocoder = torch.jit.load('hifigan_jit_0265.pts', map_location='cpu').to(device= self.device)        
        self.Load_Checkpoint(checkpoint_path)
        self.out_path = out_path
        self.batch_size = batch_size
        
    def Dataset_Generate(self, texts, speakers, emotion_ids):
        token_dict = yaml.load(open(self.hp.Token_Path), Loader=yaml.Loader)
        ge2e_dict = pickle.load(open(self.GE2E.Embedding_Dict_Path, 'rb'))

        return torch.utils.data.DataLoader(
            dataset= Dataset(token_dict, ge2e_dict, texts, speakers, emotion_ids),
            shuffle= False,
            collate_fn= Inference_Collater(token_dict),
            batch_size= self.batch_size,
            num_workers= 0,
            pin_memory= True
            )

    def Load_Checkpoint(self, path):   
        state_dict = torch.load(path, map_location= 'cpu')
        self.model.load_state_dict(state_dict['Model'])
        self.steps = state_dict['Steps']
        
        self.model.eval()

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

    
    @torch.no_grad()
    def Inference_Step(self, tokens, token_lengths, ge2es, emotion_ids, texts, decomposed_texts, speakers, restoring, export_files= False, start_index= 0, tag_step= False, tag_index= False):
        tokens = tokens.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)  
        ge2es = ge2es.to(self.device, non_blocking=True)      
        emotion_ids = emotion_ids.to(self.device, non_blocking=True)
                
        _, post_features, alignments, stops = self.model(
            tokens= tokens,
            token_lengths= token_lengths,
            ge2es= ge2es,
            emotions= emotion_ids,
            is_training= False
            )

        emotion_ids = torch.stack([emotion_ids[index] for index in restoring], dim= 0)
        texts = [texts[index] for index in restoring]
        decomposed_texts = [decomposed_texts[index] for index in restoring]
        speakers = [speakers[index] for index in restoring]

        stops = torch.stack([stops[index] for index in restoring], dim= 0)
        alignments = torch.stack([alignments[index] for index in restoring], dim= 0)
        post_features = torch.stack([post_features[index] for index in restoring], dim= 0)

        audios = []
        for feature, stop in zip(post_features, stops):
            index = (stop < 0.0).nonzero()
            index = max(index[0], 5) if len(index) > 0 else stop.size(0)
            if self.hp.Feature_Type == 'Mel':
                audio = self.vocoder(feature[:, :index].unsqueeze(0)).cpu().numpy()
                audios.append(audio)
            elif self.hp.Feature_Type == 'Spectrogram':
                feature = spectral_de_normalize_torch(feature[:, :index]).cpu().numpy()
                audio = griffinlim(feature)
                audios.append(audio)
        audios = [(audio / np.abs(audio).max() * 32767.5).astype(np.int16) for audio in audios]

        if export_files:
            files = []
            for index in range(post_features.size(0)):
                tags = []
                if tag_step: tags.append('Step-{}'.format(self.steps))
                tags.append('IDX_{}'.format(index + start_index))
                files.append('.'.join(tags))

            os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG').replace('\\', '/'), exist_ok= True)
            os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV').replace('\\', '/'), exist_ok= True)
            for index, (feature, alignment, stop, emotion_id, text, decomposed_text, speaker, audio, file) in enumerate(zip(
                post_features.cpu().numpy(),
                alignments.cpu().numpy(),
                torch.sigmoid(stops).cpu().numpy(),
                emotion_ids.cpu().numpy(),
                texts,
                decomposed_texts,
                speakers,
                audios,
                files
                )):
                title = 'Text: {}    Speaker: {}    Emotion: {}'.format(text if len(text) < 90 else text[:90] + '…', speaker, emotion_id)
                new_Figure = plt.figure(figsize=(20, 5 * 4), dpi=100)
                plt.subplot2grid((4, 1), (0, 0))
                plt.imshow(feature, aspect='auto', origin='lower')
                plt.title('Feature    {}'.format(title))
                plt.colorbar()
                plt.subplot2grid((4, 1), (1, 0), rowspan= 2)
                plt.imshow(alignment[:len(decomposed_text) + 2], aspect='auto', origin='lower')
                plt.title('Alignment    {}'.format(title))
                plt.yticks(
                    range(len(decomposed_text) + 2),
                    ['<S>'] + list(decomposed_text) + ['<E>'],
                    fontsize = 10
                    )
                plt.colorbar()
                plt.subplot2grid((4, 1), (3, 0))
                plt.plot(stop)
                plt.title('Stop token    {}'.format(title))
                plt.margins(x= 0)
                plt.axvline(x= np.argmax(stop < 0.5) if any(stop < 0.5) else stop.shape[0], linestyle='--', linewidth=1)
                plt.axhline(y=0.5, linestyle='--', linewidth=1)
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG', '{}.png'.format(file)).replace('\\', '/'))
                plt.close(new_Figure)
                
                wavfile.write(
                    os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV', '{}.wav'.format(file)).replace('\\', '/'),
                    self.hp.Sound.Sample_Rate,
                    audio
                    )

        return post_features, stops, audios

    def Inference_Epoch(self, texts, speakers, emotion_ids, export_files= False, use_tqdm= True):
        dataloader = self.Dataset_Generate(texts, speakers, emotion_ids)
        if use_tqdm:
            dataloader = tqdm(
                dataloader,
                desc='[Inference]',
                total= math.ceil(len(dataloader.dataset) / self.batch_size)
                )
        
        export_features, export_stops, export_audios = [], [], []
        for step, (tokens, token_lengths, ge2es, emotion_ids, texts, decomposed_texts, speakers, restoring) in enumerate(dataloader):
            features, stops, audios = self.Inference_Step(tokens, token_lengths, ge2es, emotion_ids, texts, decomposed_texts, speakers, restoring, export_files= export_files, start_index= step * self.batch_size)
            export_features.append(features)
            export_stops.append(stops)
            export_audios.append(audios)

        export_features = [feature for features in export_features for feature in features]
        export_stops = [stop for stops in export_stops for stop in stops]
        export_audios = [audio for audios in export_audios for audio in audios]
        
        return export_features, export_stops, export_audios

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-checkpoint', '--checkpoint', type= str, required= True)
    argParser.add_argument('-outdir', '--outdir', type= str, required= True)
    argParser.add_argument('-label', '--label', type= str, required= True)
    #argParser.add_argument('-text', '--text', type= str, required= True)
    argParser.add_argument('-text', '--text', type= str, default= '학교종이 땡땡땡 어서모이자 선생님이 우리를 기다리신다.')
    argParser.add_argument('-speaker', '--speaker', type= int, default= 0)
    argParser.add_argument('-ref', '--ref', type= str, required= False)
    argParser.add_argument('-scale', '--scale', type= float, required= True)
    argParser.add_argument('-batch', '--batch', default= 1, type= int)
    argParser.add_argument('-gpu', '--gpu', default= -1, type= int)
    args = argParser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES']= str(args.gpu)

    # Data for Console    
    new_Inferencer = Inferencer(checkpoint_path= args.checkpoint, out_path= args.outdir, batch_size= args.batch)

    labels = [args.label]
    texts = [args.text]
    length_Scales = [args.scale]
    speakers = [args.speaker]
    refs = [args.ref]
    
    new_Inferencer.Inference_Epoch(
        labels= labels,
        texts= texts,
        length_scales= length_Scales,
        speaker= speakers,
        wavs_for_speaker= refs,
        wavs_for_prosody= refs,
        wavs_for_pitch= refs
        )

    new_Inferencer.model

    
# python Inference.py -checkpoint /data/models/Glow_TTS_Kor/SR24K.PE.KSSEMOYUA/Checkpoint/S_74000.pt -outdir ./test -label test -ref ./Wav_for_Inference/YUA_SAD.wav -scale 1.1 -gpu 2