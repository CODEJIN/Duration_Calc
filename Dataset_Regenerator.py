import pickle, base64, requests, os
from scipy.io import wavfile
from tqdm import tqdm

from Inference import Inferencer

def Get_Audios(texts, speaker_ids, emotion_ids):
    inferencer = Inferencer(
        hp_path = 'Hyper_Parameters.yaml',
        checkpoint_path = '/data/results/Tacotron_SE/Checkpoint/S_38293.pt',
        out_path = './',
        batch_size= 96
        )
    post_features, stops, audios = inferencer.Inference_Epoch(texts, speaker_ids, emotion_ids)
    del post_features, stops

    return audios, inferencer.hp.Sound.Sample_Rate

def Get_ASR_Results(audios, sample_rate):
    url = 'http://10.130.198.16:8003/inference/ko'
    headers = {'Content-Type': 'application/json; charset=utf-8'}

    texts = []
    for audio in tqdm(audios):
        wavfile.write(
            os.path.join('.temp.wav').replace('\\', '/'),
            sample_rate,
            audio
            )
        with open('.temp.wav', 'rb') as f:
            payload = {
                'record_id': '0',
                'record' : base64.b64encode(f.read()).decode(),
                'language_model_weight': 0.5
                }
        response = requests.request('post', url, json=payload)
        
        texts.append(response.json()['result'][0]['Answer'])
    
    return texts

def Compare(texts1, texts2):
    comparisions = []
    for text1, text2 in tqdm(zip(texts1, texts2)):
        text1 = text1.replace(' ', '').replace(',', '').replace('!', '').replace('?', '').replace('.', '')
        text2 = text2.replace(' ', '').replace(',', '').replace('!', '').replace('?', '').replace('.', '')
        comparisions.append(text1 == text2)

    return comparisions

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']= '0' # Left space

    speaker_dict = {
        8: 'Bellona',
        9: 'Carrot',
        10: 'Cerise', 
        16: 'Elena',  
        17: 'JPS',   
        19: 'Lilias', 
        30: 'Ray',    
        31: 'Ring',   
        32: 'Sez',   
        33: 'YUA',    
        }
    emotion_dict = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad'}
    
    for speaker_id in [32,33]: # [8,9,10,16,17,19,30,31,32,33]:
        texts = []
        for root, _, files in os.walk('/rawdata/Emotion/main/lmy/transcript'):
            for file in files:
                texts.append(open(os.path.join(root, file), encoding= 'utf-8-sig').readlines()[0].strip())
        emotion_ids = [2]

        texts, speaker_ids, emotion_ids = zip(*[
            (text, speaker_id, emotion_id)
            for text in texts
            for emotion_id in emotion_ids
            ])

        audios, sample_rate = Get_Audios(texts, speaker_ids, emotion_ids)
        asr_texts = Get_ASR_Results(audios, sample_rate)
        comparisons = Compare(texts, asr_texts)

        print('Exported patterns: {}'.format(sum(comparisons)))

        os.makedirs('./Generated_Patterns/wav', exist_ok= True)
        export_scripts = ['\t'.join(['Path', 'Script', 'Speaker', 'Emotion'])]
        for index, (text, speaker_id, emotion_id, audio, comparison) in enumerate(zip(texts, speaker_ids, emotion_ids, audios, comparisons)):
            if not comparison:
                continue
            wavfile.write(
                os.path.join('./Generated_Patterns/wav/Generated_{:08d}.{}.{}.wav'.format(index, speaker_dict[speaker_id], emotion_dict[emotion_id])).replace('\\', '/'),
                sample_rate,
                audio
                )
            export_scripts.append('\t'.join(['wav/Generated_{:08d}.{}.{}.wav'.format(index, speaker_dict[speaker_id], emotion_dict[emotion_id]), text, speaker_dict[speaker_id], emotion_dict[emotion_id]]))

        open('./Generated_Patterns/scripts.{}.{}.txt'.format(speaker_dict[speaker_id], emotion_dict[emotion_id]), 'w').write('\n'.join(export_scripts))