import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'    # This is ot prevent to be called Fortran Ctrl+C crash in Windows.
import torch
import numpy as np
import logging, yaml, os, sys, argparse, math, pickle, wandb
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from Modules import ASR
from Guided_Attention import Guided_Attention_Loss
from Datasets import Dataset, Collater, Token_Interpreter
from Radam import RAdam
from Noam_Scheduler import Modified_Noam_Scheduler
from Logger import Logger

from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from Arg_Parser import Recursive_Parse, To_Non_Recursive_Dict

import matplotlib as mpl
# 유니코드 깨짐현상 해결
mpl.rcParams['axes.unicode_minus'] = False
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

# torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, hp_path, steps= 0):
        self.hp_Path = hp_path
        self.gpu_id = int(os.getenv('RANK', '0'))
        self.num_gpus = int(os.getenv("WORLD_SIZE", '1'))
        
        self.hp = Recursive_Parse(yaml.load(
            open(self.hp_Path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(self.gpu_id))
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(self.gpu_id)
        
        self.steps = steps

        self.Dataset_Generate()
        self.Model_Generate()
        self.Load_Checkpoint()
        self._Set_Distribution()

        self.scalar_dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }

        if self.gpu_id == 0:
            self.writer_dict = {
                'Train': Logger(os.path.join(self.hp.Log_Path, 'Train')),
                'Evaluation': Logger(os.path.join(self.hp.Log_Path, 'Evaluation')),
                }

        if self.hp.Weights_and_Biases.Use:
            wandb.init(
                project= self.hp.Weights_and_Biases.Project,
                entity= self.hp.Weights_and_Biases.Entity,
                name= self.hp.Weights_and_Biases.Name,
                config= To_Non_Recursive_Dict(self.hp)
                )
            wandb.watch(self.model)

    def Dataset_Generate(self):
        token_dict = yaml.load(open(self.hp.Token_Path), Loader=yaml.Loader)
        language_dict = yaml.load(open(self.hp.Language_Info_Path), Loader=yaml.Loader)

        train_dataset = Dataset(
            token_dict= token_dict,
            language_dict= language_dict,
            pattern_path= self.hp.Train.Train_Pattern.Path,
            metadata_file= self.hp.Train.Train_Pattern.Metadata_File,
            feature_type= self.hp.Feature_Type,
            accumulated_dataset_epoch= self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch,
            augmentation_ratio= self.hp.Train.Train_Pattern.Augmentation_Ratio
            )
        eval_dataset = Dataset(
            token_dict= token_dict,
            language_dict= language_dict,
            pattern_path= self.hp.Train.Eval_Pattern.Path,
            metadata_file= self.hp.Train.Eval_Pattern.Metadata_File,
            feature_type= self.hp.Feature_Type,
            )

        if self.gpu_id == 0:
            logging.info('The number of train patterns = {}.'.format(len(train_dataset) // self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch))
            logging.info('The number of development patterns = {}.'.format(len(eval_dataset)))

        collater = Collater(token_dict= token_dict)

        self.dataloader_dict = {}
        self.dataloader_dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_dataset,
            sampler= torch.utils.data.DistributedSampler(train_dataset, shuffle= True) \
                     if self.hp.Use_Multi_GPU else \
                     torch.utils.data.RandomSampler(train_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Eval'] = torch.utils.data.DataLoader(
            dataset= eval_dataset,
            sampler= torch.utils.data.DistributedSampler(eval_dataset, shuffle= True) \
                     if self.num_gpus > 1 else \
                     torch.utils.data.RandomSampler(eval_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )

        self.token_interpreter = Token_Interpreter(token_dict= token_dict)

    def Model_Generate(self):
        self.model = ASR(self.hp).to(self.device)
        self.criterion_dict = {
            'CEL': torch.nn.CrossEntropyLoss().to(self.device),
            'GAL': Guided_Attention_Loss(),
            }
        self.optimizer = RAdam(
            params= self.model.parameters(),
            lr= self.hp.Train.Learning_Rate.Initial,
            betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
            eps= self.hp.Train.ADAM.Epsilon,
            weight_decay= self.hp.Train.Weight_Decay
            )
        self.scheduler = Modified_Noam_Scheduler(
            optimizer= self.optimizer,
            base= self.hp.Train.Learning_Rate.Base
            )
        
        self.scaler = torch.cuda.amp.GradScaler(enabled= self.hp.Use_Mixed_Precision)

        if self.gpu_id == 0:
            logging.info(self.model)


    def Train_Step(self, features, feature_lengths, tokens, token_lengths, languages):
        loss_dict = {}
        features = features.to(self.device, non_blocking=True)
        feature_lengths = feature_lengths.to(self.device, non_blocking=True)
        tokens = tokens.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)
        languages = languages.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
            predictions, alignments = self.model(
                features= features,
                feature_lengths= feature_lengths,
                languages= languages,
                tokens= tokens,
                )

            loss_dict['ASR'] = self.criterion_dict['CEL'](predictions, tokens)
            loss_dict['Guided_Attention'] = self.criterion_dict['GAL'](alignments, token_lengths, feature_lengths)
            loss_dict['Total'] = loss_dict['ASR'] + loss_dict['Guided_Attention']

        self.optimizer.zero_grad()
        self.scaler.scale(loss_dict['Total']).backward()
        if self.hp.Train.Gradient_Norm > 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model.parameters(),
                max_norm= self.hp.Train.Gradient_Norm
                )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Train']['Loss/{}'.format(tag)] += loss

    def Train_Epoch(self):
        for features, feature_lengths, tokens, token_lengths, languages in self.dataloader_dict['Train']:
            self.Train_Step(features, feature_lengths, tokens, token_lengths, languages)

            if self.steps % self.hp.Train.Checkpoint_Save_Interval == 0:
                self.Save_Checkpoint()

            if self.steps % self.hp.Train.Logging_Interval == 0 and self.gpu_id == 0:
                self.scalar_dict['Train'] = {
                    tag: loss / self.hp.Train.Logging_Interval
                    for tag, loss in self.scalar_dict['Train'].items()
                    }
                self.scalar_dict['Train']['Learning_Rate'] = self.scheduler.get_last_lr()[0]
                self.writer_dict['Train'].add_scalar_dict(self.scalar_dict['Train'], self.steps)
                if self.hp.Weights_and_Biases.Use:
                    wandb.log(
                        data= {
                            f'Train.{key}': value
                            for key, value in self.scalar_dict['Train'].items()
                            },
                        step= self.steps,
                        commit= self.steps % self.hp.Train.Evaluation_Interval != 0
                        )
                self.scalar_dict['Train'] = defaultdict(float)

            if self.steps % self.hp.Train.Evaluation_Interval == 0:
                self.Evaluation_Epoch()
            
            if self.steps >= self.hp.Train.Max_Step:
                return


    @torch.no_grad()
    def Evaluation_Step(self, features, feature_lengths, tokens, token_lengths, languages):
        loss_dict = {}
        features = features.to(self.device, non_blocking=True)
        feature_lengths = feature_lengths.to(self.device, non_blocking=True)
        tokens = tokens.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)
        languages = languages.to(self.device, non_blocking=True)

        predictions, alignments = self.model(
            features= features,
            feature_lengths= feature_lengths,
            languages= languages,
            tokens= tokens
            )

        loss_dict['ASR'] = self.criterion_dict['CEL'](predictions, tokens)
        loss_dict['Guided_Attention'] = self.criterion_dict['GAL'](alignments, token_lengths, feature_lengths)
        loss_dict['Total'] = loss_dict['ASR'] + loss_dict['Guided_Attention']

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Evaluation']['Loss/{}'.format(tag)] += loss

        return predictions, alignments

    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation in GPU {}.'.format(self.steps, self.gpu_id))

        self.model.eval()

        for step, (features, feature_lengths, tokens, token_lengths, languages) in tqdm(
            enumerate(self.dataloader_dict['Eval'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataloader_dict['Eval'].dataset) / self.hp.Train.Batch_Size / self.num_gpus)
            ):
            predictions, alignments = self.Evaluation_Step(features, feature_lengths, tokens, token_lengths, languages)

        if self.gpu_id == 0:
            self.scalar_dict['Evaluation'] = {
                tag: loss / step
                for tag, loss in self.scalar_dict['Evaluation'].items()
                }
            self.writer_dict['Evaluation'].add_scalar_dict(self.scalar_dict['Evaluation'], self.steps)
            self.writer_dict['Evaluation'].add_histogram_model(self.model, 'ASR', self.steps, delete_keywords=['layer_Dict', 'layer'])
        
            index = np.random.randint(0, tokens.size(0))
            image_dict = {
                'Feature': (features[index, :, :feature_lengths[index]].cpu().numpy(), None, 'auto', None, None, None),
                'Alignment': (alignments[index, :feature_lengths[index], :token_lengths[index]].cpu().numpy().T, None, 'auto', None, None, None)
                }
            self.writer_dict['Evaluation'].add_image_dict(image_dict, self.steps)

            text_dict = {
                'Attention': self.token_interpreter.Kor_Compose(self.token_interpreter.Clean(self.token_interpreter.Interpreter(
                    predictions[index].argmax(dim= 0).cpu().numpy()
                    ))),
                'Target': self.token_interpreter.Kor_Compose(self.token_interpreter.Clean(self.token_interpreter.Interpreter(
                    tokens[index].cpu().numpy()
                    ))),
                }
            self.writer_dict['Evaluation'].add_text_dict(text_dict, self.steps)

            if self.hp.Weights_and_Biases.Use:
                wandb.log(
                    data= {
                        f'Evaluation.{key}': value
                        for key, value in self.scalar_dict['Evaluation'].items()
                        },
                    step= self.steps,
                    commit= False
                    )
                    
                results_table = wandb.Table(columns=['Target', 'Prediction'])
                for target, prediction in zip(tokens, predictions):
                    results_table.add_data(
                        self.token_interpreter.Kor_Compose(self.token_interpreter.Clean(self.token_interpreter.Interpreter(
                            target.cpu().numpy()
                            ))),
                        self.token_interpreter.Kor_Compose(self.token_interpreter.Clean(self.token_interpreter.Interpreter(
                            prediction.argmax(dim= 0).cpu().numpy()
                            )))
                        )
                wandb.log(
                    data= {
                        'Evaluation.Transcription': results_table,
                        'Evaluation.Feature': wandb.Image(features[index, :, :feature_lengths[index]].cpu().numpy()),
                        'Evaluation.Alignment': wandb.Image(alignments[index, :feature_lengths[index], :token_lengths[index]].cpu().numpy().T),
                        },
                    step= self.steps,
                    commit= True
                    )
                    
        self.scalar_dict['Evaluation'] = defaultdict(float)

        self.model.train()

    def Load_Checkpoint(self):
        if self.steps == 0:
            paths = [
                os.path.join(root, file).replace('\\', '/')
                for root, _, files in os.walk(self.hp.Checkpoint_Path)
                for file in files
                if os.path.splitext(file)[1] == '.pt'
                ]
            if len(paths) > 0:
                path = max(paths, key = os.path.getctime)
            else:
                return  # Initial training
        else:
            path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        state_dict = torch.load(path, map_location= 'cpu')
        self.model.load_state_dict(state_dict['Model'])
        self.optimizer.load_state_dict(state_dict['Optimizer'])
        self.scheduler.load_state_dict(state_dict['Scheduler'])
        self.steps = state_dict['Steps']

        logging.info('Checkpoint loaded at {} steps in GPU {}.'.format(self.steps, self.gpu_id))

    def Save_Checkpoint(self):
        if self.gpu_id != 0:
            return

        os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
        state_Dict = {
            'Model': self.model.state_dict(),
            'Optimizer': self.optimizer.state_dict(),
            'Scheduler': self.scheduler.state_dict(),
            'Steps': self.steps
            }
        checkpoint_path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        torch.save(state_Dict, checkpoint_path)

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))

        if all([
            self.hp.Weights_and_Biases.Use,
            self.hp.Weights_and_Biases.Save_Checkpoint.Use,
            self.steps % self.hp.Weights_and_Biases.Save_Checkpoint.Interval == 0
            ]):
            wandb.save(checkpoint_path)


    def _Set_Distribution(self):
        if self.num_gpus > 1:
            self.model = apply_gradient_allreduce(self.model)

    def Train(self):
        hp_path = os.path.join(self.hp.Checkpoint_Path, 'Hyper_Parameters.yaml').replace('\\', '/')
        if not os.path.exists(hp_path):
            os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
            yaml.dump(self.hp, open(hp_path, 'w'))

        if self.steps == 0:
            self.Evaluation_Epoch()

        self.tqdm = tqdm(
            initial= self.steps,
            total= self.hp.Train.Max_Step,
            desc='[Training]'
            )

        while self.steps < self.hp.Train.Max_Step:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)
            
        self.tqdm.close()
        logging.info('Finished training.')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    argParser.add_argument('-s', '--steps', default= 0, type= int)    
    argParser.add_argument('-p', '--port', default= 54321, type= int)
    argParser.add_argument('-r', '--local_rank', default= 0, type= int)
    args = argParser.parse_args()
    
    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))
    os.environ['CUDA_VISIBLE_DEVICES'] = hp.Device

    if hp.Use_Multi_GPU:
        init_distributed(
            rank= int(os.getenv('RANK', '0')),
            num_gpus= int(os.getenv("WORLD_SIZE", '1')),
            dist_backend= 'nccl',
            dist_url= 'tcp://127.0.0.1:{}'.format(args.port)
            )
    new_Trainer = Trainer(hp_path= args.hyper_parameters, steps= args.steps)
    new_Trainer.Train()