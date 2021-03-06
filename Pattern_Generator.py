import torch
import numpy as np
import yaml, os, pickle, librosa, re, argparse
from concurrent.futures import ThreadPoolExecutor as PE
from random import shuffle
from tqdm import tqdm
import hgtk
from pysptk.sptk import rapt

from meldataset import mel_spectrogram, spectrogram, spec_energy
from Arg_Parser import Recursive_Parse


using_Extension = [x.upper() for x in ['.wav', '.m4a', '.flac']]
regex_Checker = re.compile('[가-힣,.?!\'\-\s]+')
top_db_dict = {'KSS': 35, 'Emotion': 30, 'AIHub': 30, 'Seoul': 20, 'YUA': 40, 'JPS': 40}

def Text_Filtering(text):
    remove_Letter_List = ['(', ')', '\"', '[', ']', ':', ';']
    replace_List = [('  ', ' '), (' ,', ','), ('\' ', '\'')]

    text = text.upper().strip()
    for filter in remove_Letter_List:
        text= text.replace(filter, '')
    for filter, replace_STR in replace_List:
        text= text.replace(filter, replace_STR)

    text= text.strip()
    
    if len(regex_Checker.findall(text)) != 1:
        return None
    elif text.startswith('\''):
        return None
    else:
        return regex_Checker.findall(text)[0]

def Decompose(text):
    decomposed = []
    for letter in text:
        if not hgtk.checker.is_hangul(letter):
            decomposed.append(letter)
            continue

        onset, nucleus, coda = hgtk.letter.decompose(letter)
        coda += '_'
        decomposed.extend([onset, nucleus, coda])

    return decomposed

def Pattern_Generate(
    path,
    n_fft: int,
    num_mels: int,
    sample_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int,
    center: bool= False,
    top_db= 60
    ):
    audio, _ = librosa.load(path, sr= sample_rate)
    audio = librosa.effects.trim(audio, top_db=top_db, frame_length= 512, hop_length= 256)[0]
    audio = librosa.util.normalize(audio) * 0.95
    audio = audio[:audio.shape[0] - (audio.shape[0] % hop_size)]
    spect = spectrogram(
        y= torch.from_numpy(audio).float().unsqueeze(0),
        n_fft= n_fft,
        hop_size= hop_size,
        win_size= win_size,
        center= center
        ).squeeze(0).T.numpy()
    mel = mel_spectrogram(
        y= torch.from_numpy(audio).float().unsqueeze(0),
        n_fft= n_fft,
        num_mels= num_mels,
        sampling_rate= sample_rate,
        hop_size= hop_size,
        win_size= win_size,
        fmin= fmin,
        fmax= fmax,
        center= center
        ).squeeze(0).T.numpy()

    log_f0 = np.log(rapt(
        x= audio * 32768,
        fs= sample_rate,
        hopsize= hop_size,
        ))

    energy = spec_energy(
        y= torch.from_numpy(audio).float().unsqueeze(0),
        n_fft= n_fft,
        hop_size= hop_size,
        win_size= win_size,
        center= center
        ).squeeze(0).numpy()

    if log_f0.shape[0] != mel.shape[0]:
        print(path, audio.shape[0], log_f0.shape[0], mel.shape[0])

    return audio, spect, mel, log_f0, energy

def Pattern_File_Generate(path, speaker_id, speaker, emotion_id, emotion, dataset, text, decomposed, tag='', eval= False):
    pattern_path = hp.Train.Eval_Pattern.Path if eval else hp.Train.Train_Pattern.Path

    file = '{}.{}{}.PICKLE'.format(
        speaker if dataset in speaker else '{}.{}'.format(dataset, speaker),
        '{}.'.format(tag) if tag != '' else '',
        os.path.splitext(os.path.basename(path))[0]
        ).upper()
    file = os.path.join(pattern_path, dataset, speaker, file).replace("\\", "/")

    if os.path.exists(file):
        return

    audio, spect, mel, log_f0, energy = Pattern_Generate(
        path= path,
        n_fft= hp.Sound.N_FFT,
        num_mels= hp.Sound.Mel_Dim,
        sample_rate= hp.Sound.Sample_Rate,
        hop_size= hp.Sound.Frame_Shift,
        win_size= hp.Sound.Frame_Length,
        fmin= hp.Sound.Mel_F_Min,
        fmax= hp.Sound.Mel_F_Max,
        top_db= top_db_dict[dataset] if dataset in top_db_dict.keys() else 60
        )
    new_Pattern_dict = {
        'Audio': audio.astype(np.float32),
        'Spectrogram': spect.astype(np.float32),
        'Mel': mel.astype(np.float32),
        'Log_F0': log_f0.astype(np.float32),
        'Energy': energy.astype(np.float32),
        'Speaker_ID': speaker_id,
        'Speaker': speaker,
        'Emotion_ID': emotion_id,
        'Emotion': emotion,
        'Dataset': dataset,
        'Text': text,
        'Decomposed': decomposed
        }

    os.makedirs(os.path.join(pattern_path, dataset, speaker).replace('\\', '/'), exist_ok= True)
    with open(file, 'wb') as f:
        pickle.dump(new_Pattern_dict, f, protocol=4)


def Emotion_Info_Load(path):
    '''
    ema, emb, emf, emg, emh, nea, neb, nec, ned, nee, nek, nel, nem, nen, neo
    1-100: Neutral
    101-200: Happy
    201-300: Sad
    301-400: Angry

    lmy, ava, avb, avc, avd, ada, adb, adc, add:
    all neutral
    '''
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)

    text_dict = {}
    decomposed_dict = {}
    for wav_path in paths:
        text = open(wav_path.replace('/wav/', '/transcript/').replace('.wav', '.txt'), 'r', encoding= 'utf-8-sig').readlines()[0].strip()
        text = Text_Filtering(text)
        if text is None:
            continue

        decomposed = []
        for letter in text:
            if not hgtk.checker.is_hangul(letter):
                decomposed.append(letter)
                continue

            onset, nucleus, coda = hgtk.letter.decompose(letter)
            coda += '_'
            decomposed.extend([onset, nucleus, coda])

        text_dict[wav_path] = text
        decomposed_dict[wav_path] = decomposed

    paths = list(text_dict.keys())

    speaker_dict = {
        path: path.split('/')[-3].strip().upper()
        for path in paths
        }
    
    emotion_dict = {}
    for path in paths:
        if speaker_dict[path] in ['LMY', 'AVA', 'AVB', 'AVC', 'AVD', 'ADA', 'ADB', 'ADC', 'ADD']:
            emotion_dict[path] = 'Neutral'
        elif speaker_dict[path] in ['EMA', 'EMB', 'EMF', 'EMG', 'EMH', 'NEA', 'NEB', 'NEC', 'NED', 'NEE', 'NEK', 'NEL', 'NEM', 'NEN', 'NEO']:
            index = int(os.path.splitext(os.path.basename(path))[0][-5:])
            if index > 0 and index < 101:
                emotion_dict[path] = 'Neutral'
            elif index > 100 and index < 201:
                emotion_dict[path] = 'Happy'
            elif index > 200 and index < 301:
                emotion_dict[path] = 'Sad'
            elif index > 300 and index < 401:
                emotion_dict[path] = 'Angry'
            else:
                raise NotImplementedError('Unknown emotion index: {}'.format(index))
        else:
            raise NotImplementedError('Unknown speaker: {}'.format(speaker_dict[path]))

    print('Emotion info generated: {}'.format(len(paths)))
    return paths, text_dict, decomposed_dict, speaker_dict, emotion_dict

def Yua_Info_Load(path):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)

    text_dict = {}
    decomposed_dict = {}
    emotion_dict = {}
    
    for line in open(os.path.join(path, 'scripts.txt').replace('\\', '/'), 'r', encoding= 'utf-8').readlines()[1:]:
        file, _, text, emotion = line.strip().split('\t')
        text = Text_Filtering(text)
        if text is None:
            continue

        decomposed = Decompose(text)

        text_dict[os.path.join(path, file).replace('\\', '/')] = text
        decomposed_dict[os.path.join(path, file).replace('\\', '/')] = decomposed
        emotion_dict[os.path.join(path, file).replace('\\', '/')] = emotion

    paths = list(text_dict.keys())

    speaker_dict = {
        path: 'YUA'
        for path in paths
        }

    print('YUA info generated: {}'.format(len(paths)))
    return paths, text_dict, decomposed_dict, speaker_dict, emotion_dict

def JPS_Info_Load(path):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)

    text_dict = {}
    decomposed_dict = {}
    emotion_dict = {}

    for line in open(os.path.join(path, 'scripts.txt').replace('\\', '/'), 'r', encoding= 'utf-8').readlines()[1:]:
        file, _, text, emotion = line.strip().split('\t')
        text = Text_Filtering(text)
        if text is None:
            continue

        decomposed = Decompose(text)

        text_dict[os.path.join(path, file).replace('\\', '/')] = text
        decomposed_dict[os.path.join(path, file).replace('\\', '/')] = decomposed
        emotion_dict[os.path.join(path, file).replace('\\', '/')] = emotion

    paths = list(text_dict.keys())

    speaker_dict = {
        path: 'JPS'
        for path in paths
        }

    print('JPS info generated: {}'.format(len(paths)))
    return paths, text_dict, decomposed_dict, speaker_dict, emotion_dict

def Basic_Info_Load(path, label):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)
    
    text_dict = {}
    decomposed_dict = {}
    speaker_dict = {}
    emotion_dict = {}

    for line in open(os.path.join(path, 'scripts.txt').replace('\\', '/'), 'r', encoding= 'utf-8').readlines():
        file, text, speaker, emotion = line.strip().split('\t')
        text = Text_Filtering(text)
        if text is None:
            continue

        decomposed = Decompose(text)
        text_dict[os.path.join(path, file).replace('\\', '/')] = text
        decomposed_dict[os.path.join(path, file).replace('\\', '/')] = decomposed
        speaker_dict[os.path.join(path, file).replace('\\', '/')] = speaker
        emotion_dict[os.path.join(path, file).replace('\\', '/')] = emotion

    paths = list(text_dict.keys())

    print('{} info generated: {}'.format(label, len(paths)))
    return paths, text_dict, decomposed_dict, speaker_dict, emotion_dict


def Epic7_Info_Load(path):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)
    
    text_dict = {}
    decomposed_dict = {}
    speaker_dict = {}
    emotion_dict = {}

    for line in open(os.path.join(path, 'scripts.txt').replace('\\', '/'), 'r', encoding= 'utf-8').readlines():
        file, text, speaker, emotion = line.strip().split('\t')
        text = Text_Filtering(text)
        if text is None:
            continue

        decomposed = Decompose(text)
        text_dict[os.path.join(path, file).replace('\\', '/')] = text
        decomposed_dict[os.path.join(path, file).replace('\\', '/')] = decomposed
        speaker_dict[os.path.join(path, file).replace('\\', '/')] = speaker
        emotion_dict[os.path.join(path, file).replace('\\', '/')] = emotion

    paths = list(text_dict.keys())

    print('Epic7 info generated: {}'.format(len(paths)))
    return paths, text_dict, decomposed_dict, speaker_dict, emotion_dict

def GP_Info_Load(path):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)

    text_dict = {}
    decomposed_dict = {}
    speaker_dict = {}
    emotion_dict = {}

    for line in open(os.path.join(path, 'scripts.txt').replace('\\', '/'), 'r', encoding= 'utf-8').readlines()[1:]:
        file, text, speaker, emotion = line.strip().split('\t')
        text = Text_Filtering(text)
        if text is None:
            continue

        decomposed = Decompose(text)

        text_dict[os.path.join(path, file).replace('\\', '/')] = text
        decomposed_dict[os.path.join(path, file).replace('\\', '/')] = decomposed
        speaker_dict[os.path.join(path, file).replace('\\', '/')] = speaker
        emotion_dict[os.path.join(path, file).replace('\\', '/')] = emotion

    paths = list(text_dict.keys())

    print('GP info generated: {}'.format(len(paths)))
    return paths, text_dict, decomposed_dict, speaker_dict, emotion_dict

def Youtube_Info_Load(path):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)

    text_dict = {}
    decomposed_dict = {}
    speaker_dict = {}
    emotion_dict = {}

    for line in open(os.path.join(path, 'scripts.txt').replace('\\', '/'), 'r', encoding= 'utf-8').readlines()[1:]:
        file, text, speaker, emotion = line.strip().split('\t')
        text = Text_Filtering(text)
        if text is None:
            continue

        decomposed = Decompose(text)

        text_dict[os.path.join(path, file).replace('\\', '/')] = text
        decomposed_dict[os.path.join(path, file).replace('\\', '/')] = decomposed
        speaker_dict[os.path.join(path, file).replace('\\', '/')] = speaker
        emotion_dict[os.path.join(path, file).replace('\\', '/')] = emotion

    paths = list(text_dict.keys())

    print('Youtube info generated: {}'.format(len(paths)))
    return paths, text_dict, decomposed_dict, speaker_dict, emotion_dict

def External_Info_Load(path, dataset_label= '?'):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)

    text_dict = {}
    decomposed_dict = {}
    speaker_dict = {}
    emotion_dict = {}

    for line in open(os.path.join(path, 'scripts.txt').replace('\\', '/'), 'r', encoding= 'utf-8').readlines()[1:]:
        file, text, speaker, emotion = line.strip().split('\t')
        text = Text_Filtering(text)
        if text is None:
            continue

        decomposed = Decompose(text)

        text_dict[os.path.join(path, file).replace('\\', '/')] = text
        decomposed_dict[os.path.join(path, file).replace('\\', '/')] = decomposed
        speaker_dict[os.path.join(path, file).replace('\\', '/')] = speaker
        emotion_dict[os.path.join(path, file).replace('\\', '/')] = emotion

    paths = list(text_dict.keys())

    print('{} info generated: {}'.format(dataset_label, len(paths)))
    return paths, text_dict, decomposed_dict, speaker_dict, emotion_dict


def Speaker_Index_dict_Generate(speaker_dict):
    return {
        speaker: index
        for index, speaker in enumerate(sorted(set(speaker_dict.values())))
        }

def Emotion_Index_dict_Generate(emotion_dict):
    return {
        emotion: index
        for index, emotion in enumerate(sorted(set(emotion_dict.values())))
        }

def Split_Eval(paths, eval_ratio= 0.001, min_Eval= 1):
    shuffle(paths)
    index = max(int(len(paths) * eval_ratio), min_Eval)
    return paths[index:], paths[:index]

def Metadata_Generate(speaker_index_dict, emotion_index_dict, eval= False):
    pattern_path = hp.Train.Eval_Pattern.Path if eval else hp.Train.Train_Pattern.Path
    metadata_File = hp.Train.Eval_Pattern.Metadata_File if eval else hp.Train.Train_Pattern.Metadata_File

    log_f0_dict = {}
    energy_dict = {}

    new_Metadata_dict = {
        'N_FFT': hp.Sound.N_FFT,
        'Mel_Dim': hp.Sound.Mel_Dim,
        'Frame_Shift': hp.Sound.Frame_Shift,
        'Frame_Length': hp.Sound.Frame_Length,
        'Sample_Rate': hp.Sound.Sample_Rate,
        'File_List': [],
        'Audio_Length_Dict': {},
        'Spectrogram_Length_Dict': {},
        'Mel_Length_Dict': {},
        'F0_Length_Dict': {},
        'Energy_Length_Dict': {},
        'Speaker_ID_Dict': {},
        'Speaker_Dict': {},
        'Emotion_ID_Dict': {},
        'Emotion_Dict': {},
        'Dataset_Dict': {},
        'File_List_by_Speaker_Dict': {},
        'Text_Length_Dict': {},
        'ID_Reference': {'Speaker': speaker_index_dict, 'Emotion': emotion_index_dict}
        }

    files_TQDM = tqdm(
        total= sum([len(files) for root, _, files in os.walk(pattern_path)]),
        desc= 'Eval_Pattern' if eval else 'Train_Pattern'
        )

    for root, _, files in os.walk(pattern_path):
        for file in files:
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_dict = pickle.load(f)

            file = os.path.join(root, file).replace("\\", "/").replace(pattern_path, '').lstrip('/')

            try:
                if not all([
                    key in pattern_dict.keys()
                    for key in ('Audio', 'Spectrogram', 'Mel', 'Log_F0', 'Energy', 'Speaker_ID', 'Speaker', 'Emotion_ID', 'Emotion', 'Dataset', 'Text', 'Decomposed')
                    ]):
                    continue
                new_Metadata_dict['Audio_Length_Dict'][file] = pattern_dict['Audio'].shape[0]
                new_Metadata_dict['Spectrogram_Length_Dict'][file] = pattern_dict['Spectrogram'].shape[0]
                new_Metadata_dict['Mel_Length_Dict'][file] = pattern_dict['Mel'].shape[0]
                new_Metadata_dict['F0_Length_Dict'][file] = pattern_dict['Log_F0'].shape[0]
                new_Metadata_dict['Energy_Length_Dict'][file] = pattern_dict['Energy'].shape[0]
                new_Metadata_dict['Speaker_ID_Dict'][file] = pattern_dict['Speaker_ID']
                new_Metadata_dict['Emotion_ID_Dict'][file] = pattern_dict['Emotion_ID']
                new_Metadata_dict['Speaker_Dict'][file] = pattern_dict['Speaker']
                new_Metadata_dict['Emotion_Dict'][file] = pattern_dict['Emotion']
                new_Metadata_dict['Dataset_Dict'][file] = pattern_dict['Dataset']
                new_Metadata_dict['File_List'].append(file)
                if not pattern_dict['Speaker'] in new_Metadata_dict['File_List_by_Speaker_Dict'].keys():
                    new_Metadata_dict['File_List_by_Speaker_Dict'][pattern_dict['Speaker']] = []
                new_Metadata_dict['File_List_by_Speaker_Dict'][pattern_dict['Speaker']].append(file)
                new_Metadata_dict['Text_Length_Dict'][file] = len(pattern_dict['Text'])

                if not pattern_dict['Speaker_ID'] in log_f0_dict.keys():
                    log_f0_dict[pattern_dict['Speaker_ID']] = []
                if not pattern_dict['Speaker_ID'] in energy_dict.keys():
                    energy_dict[pattern_dict['Speaker_ID']] = []
                log_f0_dict[pattern_dict['Speaker_ID']].append(pattern_dict['Log_F0'])
                energy_dict[pattern_dict['Speaker_ID']].append(pattern_dict['Energy'])
            except:
                print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))
            files_TQDM.update(1)

    with open(os.path.join(pattern_path, metadata_File.upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Metadata_dict, f, protocol= 4)

    if not eval:
        log_f0_info_dict = {}
        for speaker_id, log_f0_list in log_f0_dict.items():
            log_f0 = np.hstack(log_f0_list)
            log_f0_info_dict[speaker_id] = {
                'Mean': np.ma.masked_invalid(log_f0).mean().item(),
                'Std': np.ma.masked_invalid(log_f0).std().item()
                }
        yaml.dump(
            log_f0_info_dict,
            open(hp.Log_F0_Info_Path, 'w')
            )

        energy_info_dict = {}
        for speaker_id, energy_list in energy_dict.items():
            energy = np.hstack(energy_list)
            energy_info_dict[speaker_id] = {
                'Mean': np.ma.masked_invalid(energy).mean().item(),
                'Std': np.ma.masked_invalid(energy).std().item()
                }
        yaml.dump(
            energy_info_dict,
            open(hp.Energy_Info_Path, 'w')
            )

    print('Metadata generate done.')

def Token_dict_Generate():
    tokens = \
        list(hgtk.letter.CHO) + \
        list(hgtk.letter.JOONG) + \
        ['{}_'.format(x) for x in hgtk.letter.JONG] + \
        [',', '.', '?', '!', '\'', '-', ' ']
    
    os.makedirs(os.path.dirname(hp.Token_Path), exist_ok= True)    
    yaml.dump(
        {token: index for index, token in enumerate(['<S>', '<E>'] + sorted(tokens))},
        open(hp.Token_Path, 'w')
        )

    return {token: index for index, token in enumerate(['<S>', '<E>'] + sorted(tokens))}

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-hp", "--hyper_parameters", required=True, type= str)
    argParser.add_argument("-emo", "--emotion_path", required=False)
    argParser.add_argument("-yua", "--yua_path", required=False)
    argParser.add_argument("-jps", "--jps_path", required=False)
    argParser.add_argument("-selectstar", "--selectstar_path", required=False)
    argParser.add_argument("-lostark", "--lostark_path", required=False)
    argParser.add_argument("-sea", "--sea_path", required=False)
    argParser.add_argument("-epic7", "--epic7_path", required=False)
    argParser.add_argument("-gp", "--gp_path", required=False)
    argParser.add_argument("-youtube", "--youtube_path", required=False)
    argParser.add_argument("-gcp", "--gcp_path", required=False)
    argParser.add_argument("-clova", "--clova_path", required=False)
    argParser.add_argument("-maum", "--maum_path", required=False)
    argParser.add_argument("-vd", "--voice_drama_path", required=False)

    argParser.add_argument("-evalr", "--eval_ratio", default= 0.001, type= float)
    argParser.add_argument("-evalm", "--eval_min", default= 1, type= int)
    argParser.add_argument("-mw", "--max_worker", default= 2, required=False, type= int)

    args = argParser.parse_args()

    global hp
    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))

    train_paths, eval_paths = [], []
    text_dict = {}
    decomposed_dict = {}
    speaker_dict = {}
    emotion_dict = {}
    dataset_dict = {}
    tag_dict = {}

    if not args.emotion_path is None:
        emotion_paths, emotion_text_dict, emotion_decomposed_dict, emotion_speaker_dict, emotion_emotion_dict = Emotion_Info_Load(path= args.emotion_path)
        emotion_paths = Split_Eval(emotion_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(emotion_paths[0])
        eval_paths.extend(emotion_paths[1])
        text_dict.update(emotion_text_dict)
        decomposed_dict.update(emotion_decomposed_dict)
        speaker_dict.update(emotion_speaker_dict)
        emotion_dict.update(emotion_emotion_dict)
        dataset_dict.update({path: 'Emotion' for paths in emotion_paths for path in paths})
        tag_dict.update({path: '' for paths in emotion_paths for path in paths})

    if not args.yua_path is None:
        yua_paths, yua_text_dict, yua_decomposed_dict, yua_speaker_dict, yua_emotion_dict = Yua_Info_Load(path= args.yua_path)
        yua_paths = Split_Eval(yua_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(yua_paths[0])
        eval_paths.extend(yua_paths[1])
        text_dict.update(yua_text_dict)
        decomposed_dict.update(yua_decomposed_dict)
        speaker_dict.update(yua_speaker_dict)
        emotion_dict.update(yua_emotion_dict)
        dataset_dict.update({path: 'YUA' for paths in yua_paths for path in paths})
        tag_dict.update({path: '' for paths in yua_paths for path in paths})
    
    if not args.jps_path is None:
        jps_paths, jps_text_dict, jps_decomposed_dict, jps_speaker_dict, jps_emotion_dict = JPS_Info_Load(path= args.jps_path)
        jps_paths = Split_Eval(jps_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(jps_paths[0])
        eval_paths.extend(jps_paths[1])
        text_dict.update(jps_text_dict)
        decomposed_dict.update(jps_decomposed_dict)
        speaker_dict.update(jps_speaker_dict)
        emotion_dict.update(jps_emotion_dict)
        dataset_dict.update({path: 'JPS' for paths in jps_paths for path in paths})
        tag_dict.update({path: '' for paths in jps_paths for path in paths})

    if not args.selectstar_path is None:
        selectstar_paths, selectstar_text_dict, selectstar_decomposed_dict, selectstar_speaker_dict, selectstar_emotion_dict = Basic_Info_Load(
            path= args.selectstar_path,
            label= 'SelectStar'
            )
        selectstar_paths = Split_Eval(selectstar_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(selectstar_paths[0])
        eval_paths.extend(selectstar_paths[1])
        text_dict.update(selectstar_text_dict)
        decomposed_dict.update(selectstar_decomposed_dict)
        speaker_dict.update(selectstar_speaker_dict)
        emotion_dict.update(selectstar_emotion_dict)
        dataset_dict.update({path: 'SelectStar' for paths in selectstar_paths for path in paths})
        tag_dict.update({path: '' for paths in selectstar_paths for path in paths})

    if not args.lostark_path is None:
        lostark_paths, lostark_text_dict, lostark_decomposed_dict, lostark_speaker_dict, lostark_emotion_dict = Basic_Info_Load(
            path= args.lostark_path,
            label= 'LostArk'
            )
        lostark_paths = Split_Eval(lostark_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(lostark_paths[0])
        eval_paths.extend(lostark_paths[1])
        text_dict.update(lostark_text_dict)
        decomposed_dict.update(lostark_decomposed_dict)
        speaker_dict.update(lostark_speaker_dict)
        emotion_dict.update(lostark_emotion_dict)
        dataset_dict.update({path: 'LostArk' for paths in lostark_paths for path in paths})
        tag_dict.update({path: '' for paths in lostark_paths for path in paths})

    if not args.sea_path is None:
        sea_paths, sea_text_dict, sea_decomposed_dict, sea_speaker_dict, sea_emotion_dict = Basic_Info_Load(
            path= args.sea_path,
            label= 'SEA'
            )
        sea_paths = Split_Eval(sea_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(sea_paths[0])
        eval_paths.extend(sea_paths[1])
        text_dict.update(sea_text_dict)
        decomposed_dict.update(sea_decomposed_dict)
        speaker_dict.update(sea_speaker_dict)
        emotion_dict.update(sea_emotion_dict)
        dataset_dict.update({path: 'SEA' for paths in sea_paths for path in paths})
        tag_dict.update({path: '' for paths in sea_paths for path in paths})
    
    if not args.epic7_path is None:
        epic7_paths, epic7_text_dict, epic7_decomposed_dict, epic7_speaker_dict, epic7_emotion_dict = Epic7_Info_Load(path= args.epic7_path)
        epic7_paths = Split_Eval(epic7_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(epic7_paths[0])
        eval_paths.extend(epic7_paths[1])
        text_dict.update(epic7_text_dict)
        decomposed_dict.update(epic7_decomposed_dict)
        speaker_dict.update(epic7_speaker_dict)
        emotion_dict.update(epic7_emotion_dict)
        dataset_dict.update({path: 'Epic7' for paths in epic7_paths for path in paths})
        tag_dict.update({path: '' for paths in epic7_paths for path in paths})


    if not args.gp_path is None:
        gp_paths, gp_text_dict, gp_decomposed_dict, gp_speaker_dict, gp_emotion_dict = GP_Info_Load(path= args.gp_path)
        gp_paths = Split_Eval(gp_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(gp_paths[0])
        eval_paths.extend(gp_paths[1])
        text_dict.update(gp_text_dict)
        decomposed_dict.update(gp_decomposed_dict)
        speaker_dict.update(gp_speaker_dict)
        emotion_dict.update(gp_emotion_dict)
        dataset_dict.update({path: 'GP' for paths in gp_paths for path in paths})
        tag_dict.update({path: '' for paths in gp_paths for path in paths})

    if not args.youtube_path is None:
        youtube_paths, youtube_text_dict, youtube_decomposed_dict, youtube_speaker_dict, youtube_emotion_dict = Youtube_Info_Load(path= args.youtube_path)
        youtube_paths = Split_Eval(youtube_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(youtube_paths[0])
        eval_paths.extend(youtube_paths[1])
        text_dict.update(youtube_text_dict)
        decomposed_dict.update(youtube_decomposed_dict)
        speaker_dict.update(youtube_speaker_dict)
        emotion_dict.update(youtube_emotion_dict)
        dataset_dict.update({path: 'Youtube' for paths in youtube_paths for path in paths})
        tag_dict.update({path: '' for paths in youtube_paths for path in paths})


    if not args.gcp_path is None:
        gcp_paths, gcp_text_dict, gcp_decomposed_dict, gcp_speaker_dict, gcp_emotion_dict = External_Info_Load(path= args.gcp_path, dataset_label= 'GCP')
        gcp_paths = Split_Eval(gcp_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(gcp_paths[0])
        eval_paths.extend(gcp_paths[1])
        text_dict.update(gcp_text_dict)
        decomposed_dict.update(gcp_decomposed_dict)
        speaker_dict.update(gcp_speaker_dict)
        emotion_dict.update(gcp_emotion_dict)
        dataset_dict.update({path: 'GCP' for paths in gcp_paths for path in paths})
        tag_dict.update({path: '' for paths in gcp_paths for path in paths})

    if not args.clova_path is None:
        clova_paths, clova_text_dict, clova_decomposed_dict, clova_speaker_dict, clova_emotion_dict = External_Info_Load(path= args.clova_path, dataset_label= 'Clova')
        clova_paths = Split_Eval(clova_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(clova_paths[0])
        eval_paths.extend(clova_paths[1])
        text_dict.update(clova_text_dict)
        decomposed_dict.update(clova_decomposed_dict)
        speaker_dict.update(clova_speaker_dict)
        emotion_dict.update(clova_emotion_dict)
        dataset_dict.update({path: 'Clova' for paths in clova_paths for path in paths})
        tag_dict.update({path: '' for paths in clova_paths for path in paths})

    if not args.maum_path is None:
        maum_paths, maum_text_dict, maum_decomposed_dict, maum_speaker_dict, maum_emotion_dict = External_Info_Load(path= args.maum_path, dataset_label= 'Maum')
        maum_paths = Split_Eval(maum_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(maum_paths[0])
        eval_paths.extend(maum_paths[1])
        text_dict.update(maum_text_dict)
        decomposed_dict.update(maum_decomposed_dict)
        speaker_dict.update(maum_speaker_dict)
        emotion_dict.update(maum_emotion_dict)
        dataset_dict.update({path: 'Maum' for paths in maum_paths for path in paths})
        tag_dict.update({path: '' for paths in maum_paths for path in paths})

    if not args.voice_drama_path is None:
        voice_drama_paths, voice_drama_text_dict, voice_drama_decomposed_dict, voice_drama_speaker_dict, voice_drama_emotion_dict = Basic_Info_Load(
            path= args.voice_drama_path,
            label= 'VD'
            )
        voice_drama_paths = Split_Eval(voice_drama_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(voice_drama_paths[0])
        eval_paths.extend(voice_drama_paths[1])
        text_dict.update(voice_drama_text_dict)
        decomposed_dict.update(voice_drama_decomposed_dict)
        speaker_dict.update(voice_drama_speaker_dict)
        emotion_dict.update(voice_drama_emotion_dict)
        dataset_dict.update({path: 'VD' for paths in voice_drama_paths for path in paths})
        tag_dict.update({path: '' for paths in voice_drama_paths for path in paths})

    if len(train_paths) == 0 or len(eval_paths) == 0:
        raise ValueError('Total info count must be bigger than 0.')

    token_dict = Token_dict_Generate()
    speaker_index_dict = Speaker_Index_dict_Generate(speaker_dict)
    emotion_index_dict = Emotion_Index_dict_Generate(emotion_dict)

    with PE(max_workers = args.max_worker) as pe:
        for _ in tqdm(
            pe.map(
                lambda params: Pattern_File_Generate(*params),
                [
                    (
                        path,
                        speaker_index_dict[speaker_dict[path]],
                        speaker_dict[path],
                        emotion_index_dict[emotion_dict[path]],
                        emotion_dict[path],
                        dataset_dict[path],
                        text_dict[path],
                        decomposed_dict[path],
                        tag_dict[path],
                        False
                        )
                    for path in train_paths
                    ]
                ),
            total= len(train_paths)
            ):
            pass
        for _ in tqdm(
            pe.map(
                lambda params: Pattern_File_Generate(*params),
                [
                    (
                        path,
                        speaker_index_dict[speaker_dict[path]],
                        speaker_dict[path],
                        emotion_index_dict[emotion_dict[path]],
                        emotion_dict[path],
                        dataset_dict[path],
                        text_dict[path],
                        decomposed_dict[path],
                        tag_dict[path],
                        True
                        )
                    for path in eval_paths
                    ]
                ),
            total= len(eval_paths)
            ):
            pass

    Metadata_Generate(speaker_index_dict, emotion_index_dict)
    Metadata_Generate(speaker_index_dict, emotion_index_dict, eval= True)

# python Pattern_Generator.py -hp ./Hyper_Parameters_No_GPYou.yaml -emo /rawdata/Emotion -jps /rawdata/JPS -yua /rawdata/Yua_201123 -epic7 /rawdata/Epic_Seven -gcp /rawdata/Dataset_Generator_from_External/External/GCP_TTS -clova /rawdata/Dataset_Generator_from_External/External/Clova -maum /rawdata/Dataset_Generator_from_External/External/Maum -evalm 3

# python Pattern_Generator.py -hp ./Hyper_Parameters.yaml  -selectstar /rawdata/SelectStar -evalm 3
# python Pattern_Generator.py -hp ./Hyper_Parameters.yaml -emo /rawdata/Emotion -jps /rawdata/JPS -yua /rawdata/Yua_201123 -epic7 /rawdata/Epic_Seven -gcp /rawdata/Dataset_Generator_from_External/External/GCP_TTS -clova /rawdata/Dataset_Generator_from_External/External/Clova -maum /rawdata/Dataset_Generator_from_External/External/Maum -selectstar /rawdata/SelectStar -lostark /rawdata/LostArk -sea /rawdata/Sea -evalm 3
# python Pattern_Generator.py -hp ./Hyper_Parameters.yaml -emo /rawdata/Emotion -jps /rawdata/JPS -yua /rawdata/Yua_201123 -epic7 /rawdata/Epic_Seven -gcp /rawdata/Dataset_Generator_from_External/External/GCP_TTS -clova /rawdata/Dataset_Generator_from_External/External/Clova -maum /rawdata/Dataset_Generator_from_External/External/Maum -selectstar /rawdata/SelectStar -lostark /rawdata/LostArk -sea /rawdata/Sea -vd /rawdata/Voice_Drama -evalm 3
