from argparse import Namespace
import torch

class ASR(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        self.encoder = Encoder(self.hp)
        self.language_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Languages,
            embedding_dim= self.hp.Encoder.Size
            )
        self.decoder = Decoder(self.hp)

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        languages: torch.Tensor,
        tokens: torch.Tensor
        ):
        encodings = self.encoder(features)
        languages = self.language_embedding(languages).unsqueeze(2)
        encodings = encodings + languages
        predictions, alignments = self.decoder(
            encodings= encodings,
            encoding_lengths= feature_lengths,
            tokens= tokens
            )

        return predictions, alignments

class Encoder(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters
        assert self.hp.Encoder.Size % 2 == 0, 'The LSTM size of text encoder must be a even number.'

        if self.hp.Feature_Type == 'Mel':
            self.feature_size = self.hp.Sound.Mel_Dim
        elif self.hp.Feature_Type == 'Spectrogram':
            self.feature_size = self.hp.Sound.N_FFT // 2 + 1

        previous_channels = self.feature_size
        self.conv = torch.nn.Sequential()
        for index in range(self.hp.Encoder.Conv.Stack):
            self.conv.add_module('Conv_{}'.format(index), Conv1d(
                in_channels= previous_channels,
                out_channels= self.hp.Encoder.Size,
                kernel_size= self.hp.Encoder.Conv.Kernel_Size,
                padding= (self.hp.Encoder.Conv.Kernel_Size - 1) // 2,
                bias= False,
                w_init_gain= 'relu'
                ))
            self.conv.add_module('BatchNorm_{}'.format(index), torch.nn.BatchNorm1d(
                num_features= self.hp.Encoder.Size
                ))
            self.conv.add_module('ReLU_{}'.format(index), torch.nn.ReLU(inplace= False))
            self.conv.add_module('Dropout_{}'.format(index), torch.nn.Dropout(
                p= self.hp.Encoder.Conv.Dropout,
                inplace= True
                ))
            previous_channels = self.hp.Encoder.Size

        self.lstm = torch.nn.LSTM(
            input_size= self.hp.Encoder.Size,
            hidden_size= self.hp.Encoder.Size // 2,
            num_layers= self.hp.Encoder.LSTM_Stack,
            batch_first= True,
            bidirectional= True
            )

    def forward(self, x: torch.Tensor):
        '''
        x: [Batch, Time]
        lengths: [Batch]
        '''
        x = self.conv(x)    # [Batch, Dim, Time]
        x = self.lstm(x.permute(0, 2, 1))[0].permute(0, 2, 1)

        return x

class Decoder(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super(Decoder, self).__init__()
        self.hp = hyper_parameters

        self.embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Tokens,
            embedding_dim= self.hp.Decoder.Size
            )
        self.prenet = torch.nn.Sequential(
            Conv1d(
                in_channels= self.hp.Decoder.Size,
                out_channels= self.hp.Decoder.Size * 4,
                kernel_size= 1,
                w_init_gain= 'relu'
                ),
            torch.nn.ReLU(),
            torch.nn.Dropout(p= self.hp.Decoder.Prenet_Dropout_Rate),
            Conv1d(
                in_channels= self.hp.Decoder.Size * 4,
                out_channels= self.hp.Decoder.Size,
                kernel_size= 1,
                w_init_gain= 'relu'
                ),
            torch.nn.ReLU(),
            torch.nn.Dropout(p= self.hp.Decoder.Prenet_Dropout_Rate),
            )

        
        self.pre_lstm = torch.nn.LSTMCell(
            input_size= self.hp.Decoder.Size + self.hp.Encoder.Size,   # encoding size == previous context size
            hidden_size= self.hp.Decoder.Pre_LSTM.Size,
            bias= True
            )
        self.pre_lstm_dropout = torch.nn.Dropout(
            p= self.hp.Decoder.Pre_LSTM.Dropout_Rate
            )

        self.attention = Location_Sensitive_Attention(
            query_size= self.hp.Decoder.Pre_LSTM.Size,
            encoding_size= self.hp.Encoder.Size,
            attention_size= self.hp.Decoder.Attention.Channels,
            attention_location_n_filters= self.hp.Decoder.Attention.Conv.Channels,
            attention_location_kernel_size= self.hp.Decoder.Attention.Conv.Kernel_Size
            )

        self.post_lstm = torch.nn.LSTMCell(
            input_size= self.hp.Decoder.Pre_LSTM.Size + self.hp.Encoder.Size,
            hidden_size= self.hp.Decoder.Post_LSTM.Size,
            bias= True
            )
        self.post_lstm_dropout = torch.nn.Dropout(
            p= self.hp.Decoder.Post_LSTM.Dropout_Rate
            )
        self.projection = Linear(
            in_features= self.hp.Decoder.Post_LSTM.Size + self.hp.Encoder.Size,
            out_features= self.hp.Tokens,
            bias= True
            )

    def forward(
        self,
        encodings: torch.Tensor,
        encoding_lengths: torch.Tensor,
        tokens: torch.Tensor
        ):
        '''
        encodings: [Batch, Enc_d, Enc_t]
        encoding_lengths: [Batch]
        tokens: [Batch, Token_d, Token_t]
        '''
        encoding_masks = Mask_Generate(encoding_lengths)

        pre_lstm_hidden, pre_lstm_cell = self.Get_LSTM_Intial_States(
            reference= encodings,
            cell_size= self.hp.Decoder.Pre_LSTM.Size
            )
        post_lstm_hidden, post_lstm_cell = self.Get_LSTM_Intial_States(
            reference= encodings,
            cell_size= self.hp.Decoder.Post_LSTM.Size
            )
        contexts, alignments = self.attention.get_initial(
            memories= encodings
            )
        cumulated_alignments = alignments
        processed_memories = self.attention.get_processed_memory(encodings)
        
        tokens = tokens[:, :-1]
        tokens = self.embedding(tokens)
        tokens = self.prenet(tokens.permute(0, 2, 1)).permute(2, 0, 1)  # [Dec_t - 1, Batch, Dec_dim]

        predictions_list, alignments_list = [], []        
        for x in tokens:
            pre_lstm_hidden, pre_lstm_cell = self.pre_lstm(
                torch.cat([x, contexts], dim= 1),  # contexts_t-1
                (pre_lstm_hidden, pre_lstm_cell)
                )
            pre_lstm_hidden = self.pre_lstm_dropout(pre_lstm_hidden)

            contexts, alignments = self.attention(
                query= pre_lstm_hidden,
                memories= encodings,
                processed_memories= processed_memories,
                previous_alignments= alignments,
                cumulated_alignments= cumulated_alignments,
                masks= encoding_masks
                )
            cumulated_alignments = cumulated_alignments + alignments

            post_lstm_hidden, post_lstm_cell = self.post_lstm(
                torch.cat([pre_lstm_hidden, contexts], dim= 1),  # contexts_t
                (post_lstm_hidden, post_lstm_cell)
                )
            post_lstm_hidden = self.post_lstm_dropout(post_lstm_hidden)

            decodings = torch.cat([post_lstm_hidden, contexts], dim= 1)

            projections = self.projection(decodings)

            predictions_list.append(projections)
            alignments_list.append(alignments)

        predictions = torch.stack(predictions_list, dim= 2)  # [Batch, Token_n, Token_t - 1]
        alignments = torch.stack(alignments_list, dim= 2)  # [Batch, Feature_t, Token_t - 1]

        return predictions, alignments


    def Get_LSTM_Intial_States(self, reference, cell_size):
        hiddens = reference.new_zeros(
            size= (reference.size(0), cell_size)
            )
        cells = reference.new_zeros(
            size= (reference.size(0), cell_size)
            )

        return hiddens, cells

    def Get_Attention_Initial_States(self, memories):
        '''
        memories: [Batch, Enc_t, Enc_dim]
        '''
        contexts = memories.new_zeros(
            size= (memories.size(0), memories.size(2))
            )
        alignments = memories.new_zeros(
            size= (memories.size(0), memories.size(1))
            )
        alignments[:, 0] = 1.0   # (Q0, M0) is 1.0
        
        return contexts, alignments

class Location_Sensitive_Attention(torch.nn.Module):
    def __init__(
        self,
        query_size,
        encoding_size,
        attention_size,
        attention_location_n_filters= 32,
        attention_location_kernel_size= 31
        ):
        super().__init__()
        self.query = Conv1d(
            in_channels= query_size,
            out_channels= attention_size,
            kernel_size= 1,
            bias= False,
            w_init_gain= 'tanh'
            )
        self.memory = Conv1d(
            in_channels= encoding_size,
            out_channels= attention_size,
            kernel_size= 1,
            bias= False,
            w_init_gain= 'tanh'
            )
        
        self.location_layer = torch.nn.Sequential(
            Conv1d(
                in_channels= 2,
                out_channels= attention_location_n_filters,
                kernel_size= attention_location_kernel_size,
                padding= (attention_location_kernel_size - 1) // 2,
                bias= False,
                ),  # [Batch, Loc_d, Enc_t]
            Conv1d(
                in_channels= attention_location_n_filters,
                out_channels= attention_size,
                kernel_size= 1,
                bias= False,
                w_init_gain= 'tanh'
                )   # [Batch, Att_d, Enc_t]
            )

        self.v = Conv1d(
            in_channels= attention_size,
            out_channels= 1,
            kernel_size= 1,
            bias= False
            )

    def get_alignment_energies(
        self,
        query,
        processed_memory,
        attention_weights_cat
        ):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, attention_dim, T_in)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """
        processed_query = self.query(query.unsqueeze(2))    # [Batch, Att_d, 1]
        processed_attention_weights = self.location_layer(attention_weights_cat)    # [Batch, Att_d, Enc_t]
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory
            ))  # [Batch, 1, Enc_t]

        return energies

    def forward(
        self,
        query,
        memories,
        processed_memories,
        previous_alignments,
        cumulated_alignments,
        masks
        ):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        attention_weights_cat = torch.stack([previous_alignments, cumulated_alignments], dim= 1)

        alignment = self.get_alignment_energies(
            query,
            processed_memories,
            attention_weights_cat
            )   # [Batch, 1, Enc_t]

        if masks is not None:
            alignment.data.masked_fill_(masks.unsqueeze(1), -torch.finfo(alignment.dtype).max)

        attention_weights = torch.nn.functional.softmax(alignment, dim=2)   # [Batch, 1, Enc_t]

        attention_context = memories @ attention_weights.permute(0, 2, 1)  # [Batch, Att_d, Enc_t] @ [Batch, 1, Enc_t] -> [Batch, Att_d, 1]

        return attention_context.squeeze(2), attention_weights.squeeze(1)

    def get_processed_memory(self, memories):
        return self.memory(memories)

    def get_initial(self, memories):
        initial_context = torch.zeros_like(memories[:, :, 0])   # [Batch, Value_d]
        initial_alignment = torch.nn.functional.one_hot(
            memories.new_zeros(memories.size(0)).long(),
            num_classes= memories.size(2)
            ).to(dtype= memories.dtype)

        return initial_context, initial_alignment



class Conv1d(torch.nn.Conv1d):
    def __init__(self, w_init_gain= 'linear', *args, **kwargs):
        self.w_init_gain = w_init_gain
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        if self.w_init_gain in ['relu', 'leaky_relu']:
            torch.nn.init.kaiming_uniform_(self.weight, nonlinearity= self.w_init_gain)
        else:
            torch.nn.init.xavier_uniform_(self.weight, gain= torch.nn.init.calculate_gain(self.w_init_gain))
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

class Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias= True, w_init_gain='linear'):
        self.w_init_gain = w_init_gain
        super(Linear, self).__init__(
            in_features= in_features,
            out_features= out_features,
            bias= bias
            )

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(
            self.weight,
            gain=torch.nn.init.calculate_gain(self.w_init_gain)
            )
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

class Lambda(torch.nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


def Mask_Generate(lengths: torch.Tensor, max_length: int= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]