"""Transformer for ASR in the SpeechBrain sytle.

Authors
* Jianyuan Zhong 2020
"""

import torch  # noqa 42
from torch import nn
from typing import Optional

from nnet import Linear
from nnet import ModuleList
from transformer_help import * 
import attention as attention
from embedding import Embedding

class TransformerClassAudio(TransformerInterface):
    """This is an implementation of transformer model for ASR.

    The architecture is based on the paper "Attention Is All You Need":
    https://arxiv.org/pdf/1706.03762.pdf

    Arguments
    ----------
    tgt_vocab: int
        Size of vocabulary.
    input_size: int
        Input feature size.
    d_model : int, optional
        Embedding dimension size.
        (default=512).
    nhead : int, optional
        The number of heads in the multi-head attention models (default=8).
    num_encoder_layers : int, optional
        The number of sub-encoder-layers in the encoder (default=6).
    num_decoder_layers : int, optional
        The number of sub-decoder-layers in the decoder (default=6).
    dim_ffn : int, optional
        The dimension of the feedforward network model (default=2048).
    n_class : int
        The number of classes in the classification layer.
    dropout : int, optional
        The dropout value (default=0.1).
    activation : torch.nn.Module, optional
        The activation function of FFN layers.
        Recommended: relu or gelu (default=relu).
    positional_encoding: str, optional
        Type of positional encoding used. e.g. 'fixed_abs_sine' for fixed absolute positional encodings.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    kernel_size: int, optional
        Kernel size in convolutional layers when Conformer is used.
    bias: bool, optional
        Whether to use bias in Conformer convolutional layers.
    encoder_module: str, optional
        Choose between Conformer and Transformer for the encoder. The decoder is fixed to be a Transformer.
    conformer_activation: torch.nn.Module, optional
        Activation module used after Conformer convolutional layers. E.g. Swish, ReLU etc. it has to be a torch Module.
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.
    max_length: int, optional
        Max length for the target and source sequence in input.
        Used for positional encodings.
    causal: bool, optional
        Whether the encoder should be causal or not (the decoder is always causal).
        If causal the Conformer convolutional layer is causal.

    Example
    -------
    >>> src = torch.rand([8, 120, 512])
    >>> tgt = torch.randint(0, 720, [8, 120])
    >>> net = TransformerASR(
    ...     720, 512, 512, 8, 1, 1, 1024, activation=torch.nn.GELU
    ... )
    >>> enc_out, dec_out = net.forward(src, tgt)
    >>> enc_out.shape
    torch.Size([8, 120, 512])
    >>> dec_out.shape
    torch.Size([8, 120, 512])
    """

    def __init__(
        self,
        tgt_vocab,
        input_size,
        d_model=512,
        d_pos_enc=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        n_class=4,
        dropout=0.1,
        activation=nn.ReLU,
        positional_encoding="fixed_abs_sine",
        normalize_before=False,
        kernel_size: Optional[int] = 31,
        bias: Optional[bool] = True,
        encoder_module: Optional[str] = "transformer",
        conformer_activation: Optional[nn.Module] = attention.Swish,
        attention_type: Optional[str] = "regularMHA",
        max_length: Optional[int] = 5000,
        causal: Optional[bool] = True,
    ):
        super().__init__(
            d_model=d_model,
            d_pos_enc=d_pos_enc,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            n_class=n_class,
            dropout=dropout,
            activation=activation,
            positional_encoding=positional_encoding,
            normalize_before=normalize_before,
            kernel_size=kernel_size,
            bias=bias,
            encoder_module=encoder_module,
            conformer_activation=conformer_activation,
            attention_type=attention_type,
            max_length=max_length,
            causal=causal,
        )

        self.custom_src_module = ModuleList(
            Linear(
                input_size=input_size,
                n_neurons=d_model,
                bias=True,
                combine_dims=False,
            ),
            torch.nn.Dropout(dropout),
        )
        # self.custom_tgt_module = ModuleList(
        #     NormalizedEmbedding(d_model, tgt_vocab)
        # )
        self.fclayer = nn.Linear(d_model, n_class)

        # reset parameters using xavier_normal_
        self._init_params()

    def forward(self, src, wav_len=None, pad_idx=0):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        pad_idx : int, optional
            The index for <pad> token (default=0).
        """

        # reshpae the src vector to [Batch, Time, Fea] is a 4d vector is given
        # print(src.shape, wav_len)
        enc = self.encode(src, wav_len)
        enc_r = enc.transpose(2,1)
        enc_r = nn.functional.avg_pool1d(enc_r, [enc_r.size()[2]], stride=1)
        enc_r = enc_r.squeeze(2)
        logits = self.fclayer(enc_r)
        logits =  nn.functional.log_softmax(logits, dim=-1)
        return enc_r, logits

    def make_masks(self, src, tgt, wav_len=None, pad_idx=0):
        """This method generates the masks for training the transformer model.

        Arguments
        ---------
        src : tensor
            The sequence to the encoder (required).
        tgt : tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).
        """
        src_key_padding_mask = None
        if wav_len is not None:
            abs_len = torch.round(wav_len * src.shape[1])
            src_key_padding_mask = (
                torch.arange(src.shape[1])[None, :].to(abs_len)
                > abs_len[:, None]
            )

        tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)

        src_mask = None
        tgt_mask = get_lookahead_mask(tgt)
        return src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask


    def encode(
        self, src, wav_len=None,
    ):
        """
        Encoder forward pass

        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        """
        # reshape the src vector to [Batch, Time, Fea] if a 4d vector is given
        if src.dim() == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        src_key_padding_mask = None
        if wav_len is not None:
            abs_len = torch.floor(wav_len * src.shape[1])
            src_key_padding_mask = (
                torch.arange(src.shape[1])[None, :].to(abs_len)
                > abs_len[:, None]
            )

        src = self.custom_src_module(src) # linear and dropout
        if self.attention_type == "RelPosMHAXL":
            pos_embs_source = self.positional_encoding(src)

        elif self.positional_encoding_type == "fixed_abs_sine":
            try: 
                src = src + self.positional_encoding(src)
            except Exception as e:
                 # print(src.shape, self.positional_encoding(src).shape)
                 print('Check the max length param in positional_encoding')
            # print(self.positional_encoding)
            # print("positional encoding: ", self.positional_encoding(src))
            pos_embs_source = None

        encoder_out, _ = self.encoder(
            src=src,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_source,
        )
        return encoder_out


    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)

class TransformerClassAudioText(TransformerInterface):
    def __init__(
        self,
        tgt_vocab,
        input_size,
        d_model=512,
        d_pos_enc=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        n_class=4,
        modalities=None,
        dropout=0.1,
        activation=nn.ReLU,
        positional_encoding="fixed_abs_sine",
        normalize_before=False,
        kernel_size: Optional[int] = 31,
        bias: Optional[bool] = True,
        encoder_module: Optional[str] = "transformer",
        conformer_activation: Optional[nn.Module] = attention.Swish,
        attention_type: Optional[str] = "regularMHA",
        max_length: Optional[int] = 5000,
        causal: Optional[bool] = True,
    ):
        super().__init__(
            d_model=d_model,
            d_pos_enc=d_pos_enc,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            n_class=n_class,
            modalities=modalities,
            dropout=dropout,
            activation=activation,
            positional_encoding=positional_encoding,
            normalize_before=normalize_before,
            kernel_size=kernel_size,
            bias=bias,
            encoder_module=encoder_module,
            conformer_activation=conformer_activation,
            attention_type=attention_type,
            max_length=max_length,
            causal=causal,
        )
        self.modalities=modalities
        self.phone_emb_size = d_model
        self.bert_emb_size = 768
        self.custom_src_module = ModuleList(
            Linear(
                input_size=input_size,
                n_neurons=d_model,
                bias=True,
                combine_dims=False,
            ),
            torch.nn.Dropout(dropout),
        )

        self.custom_src_module_phone = ModuleList(
            Linear(
                input_size=self.phone_emb_size,
                n_neurons=self.phone_emb_size,
                bias=True,
                combine_dims=False,
            ),
            torch.nn.Dropout(dropout),
        )
        self.custom_src_module_bert = ModuleList(
            Linear(
                input_size=self.bert_emb_size,
                n_neurons=self.phone_emb_size,
                bias=True,
                combine_dims=False,
            ),
            torch.nn.Dropout(dropout),
        )
        # self.custom_tgt_module = ModuleList(
        #     NormalizedEmbedding(d_model, tgt_vocab)
        # )
        if self.modalities == "a_phfeat_aligned_unaligned" or self.modalities == "a_phfeat_aligned_unaligned-2":
            self.fclayer = nn.Linear(d_model*5, n_class)
        else:
            self.fclayer = nn.Linear(d_model*3, n_class)

        if encoder_module == "transformer":
            self.textencoder = TransformerEncoder(
                nhead=nhead,
                num_layers=num_encoder_layers,
                d_ffn=d_ffn,
                d_model=self.phone_emb_size, ### CHANGED THIS 
                dropout=dropout,
                activation=activation,
                normalize_before=normalize_before,
                causal=self.causal,
                attention_type=self.attention_type,
            )
            self.wordencoder = TransformerEncoder(
                nhead=nhead,
                num_layers=num_encoder_layers,
                d_ffn=d_ffn,
                d_model=self.phone_emb_size, ### CHANGED THIS 
                dropout=dropout,
                activation=activation,
                normalize_before=normalize_before,
                causal=self.causal,
                attention_type=self.attention_type,
            )

        elif encoder_module == "conformer":
            self.textencoder = ConformerEncoder(
                nhead=nhead,
                num_layers=num_encoder_layers,
                d_ffn=d_ffn,
                d_model=d_model, #self.d_pos_enc, # changed thsi 
                dropout=dropout,
                activation=conformer_activation,
                kernel_size=kernel_size,
                bias=bias,
                causal=self.causal,
                attention_type=self.attention_type,
            )
            self.wordencoder = ConformerEncoder(
                nhead=nhead,
                num_layers=num_encoder_layers,
                d_ffn=d_ffn,
                d_model=d_model, #self.d_pos_enc,
                dropout=dropout,
                activation=conformer_activation,
                kernel_size=kernel_size,
                bias=bias,
                causal=self.causal,
                attention_type=self.attention_type,
            )
            assert (
                    normalize_before
                ), "normalize_before must be True for Conformer"

            assert (
                conformer_activation is not None
            ), "conformer_activation must not be None"
        elif encoder_module == "CNNEncoder":
                self.textencoder = CNNEncoder(d_model=d_model, d_pos_enc=d_pos_enc)
                self.wordencoder = CNNEncoder(d_model=d_model, d_pos_enc=d_pos_enc)
                
        # self.textencoder = TransformerEncoder(
        #             nhead=nhead,
        #             num_layers=num_encoder_layers,
        #             d_ffn=d_ffn,
        #             d_model=self.phone_emb_size,
        #             dropout=dropout,
        #             activation=activation,
        #             normalize_before=normalize_before,
        #             causal=self.causal,
        #             attention_type=self.attention_type,
        #         )
        self.phoneEmb = Embedding(
            num_embeddings=52,
            embedding_dim=self.phone_emb_size,
            blank_id=0
        )

        self.convmodule = torch.nn.Sequential(
                nn.Conv1d(d_pos_enc, d_pos_enc, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv1d(d_pos_enc, d_pos_enc, kernel_size=3, stride=2),
            )
        # reset parameters using xavier_normal_
        self._init_params()

    def forward(self, src, txt, trans, wav_len=None, bert_len=None, pad_idx=0):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        pad_idx : int, optional
            The index for <pad> token (default=0).
        """

        # reshpae the src vector to [Batch, Time, Fea] is a 4d vector is given
        # print("src input dimension ", src.shape)
        enc_audio, attention_lst_audio= self.audioEncode(src, wav_len) #torch.Size([3, 64, 74, 15]), what i want [3, 1, 64]
        # print("Linear and pe added: ", enc_audio.shape)
        enc_audio_r = enc_audio.transpose(2,1)
        enc_audio_r = nn.functional.avg_pool1d(enc_audio_r, [enc_audio_r.size()[2]], stride=1)
        enc_audio_r = enc_audio_r.squeeze(2)
        # print("Encoded Audio: ", enc_audio_r.shape)
        # exit()
        # print("Txt shape", txt.shape)

        txt = self.phoneEmb(txt.squeeze(2))
        # print(txt.shape)
        enc_text, attention_lst_phone = self.phoneEncode(txt, wav_len)
        enc_text_r = enc_text.transpose(2,1)
        enc_text_r = nn.functional.avg_pool1d(enc_text_r, [enc_text_r.size()[2]], stride=1)
        enc_text_r = enc_text_r.squeeze(2)
        # print("Encoded phone ", enc_text_r.shape) # [b, featsize]
        # print(enc_audio_r.shape, enc_text_r.shape)

        ########################
        # print(trans.shape)
        enc_trans, attention_lst_trans = self.wordEncode(trans, bert_len)
        enc_trans_r = enc_trans.transpose(2,1)
        enc_trans_r = nn.functional.avg_pool1d(enc_trans_r, [enc_trans_r.size()[2]], stride=1)
        enc_trans_r = enc_trans_r.squeeze(2)
        # print("Encoded phone ", enc_trans_r.shape) # [b, featsize]
        # print(attention_lst_audio)
        # import pdb 
        # pdb.set_trace()
        enc = torch.cat((enc_audio_r, enc_text_r, enc_trans_r ), dim=1) #[b, featsize*2]


        # print(enc.shape)
        # enc = torch.cat((enc_audio_r, enc_text_r ), dim=1)

        # enc = torch.cat((enc_audio_r, enc_text_r), dim=1)
        logits = self.fclayer(enc)
        logits =  nn.functional.log_softmax(logits, dim=-1)
        # print("Logits: ", logits.shape)
        return enc, logits#, (attention_lst_audio, attention_lst_phone, attention_lst_trans)

    def forward_context(self, items):#src, txt, trans, wav_len=None, bert_len=None, pad_idx=0):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        pad_idx : int, optional
            The index for <pad> token (default=0).
        """
        encitems = []
        # print("items: ", len(items))
        for i in range(3):
            src, _, _, _, _, _, txt, wav_len, \
                _, trans, bert_len, _ , _ = items[i]

            # reshpae the src vector to [Batch, Time, Fea] is a 4d vector is given
            # print(src.shape)
            enc_audio = self.audioEncode(src, wav_len) #torch.Size([3, 64, 74, 15]), what i want [3, 1, 64]
            # print(enc_audio.shape)
            enc_audio_r = enc_audio.transpose(2,1)
            enc_audio_r = nn.functional.avg_pool1d(enc_audio_r, [enc_audio_r.size()[2]], stride=1)
            enc_audio_r = enc_audio_r.squeeze(2)
            # print("Encoded Audio: ", enc_audio_r.shape)
            # print("Txt shape", txt.shape)

            txt = self.phoneEmb(txt.squeeze(2))
            # print(txt.shape)
            enc_text = self.phoneEncode(txt, wav_len)
            enc_text_r = enc_text.transpose(2,1)
            enc_text_r = nn.functional.avg_pool1d(enc_text_r, [enc_text_r.size()[2]], stride=1)
            enc_text_r = enc_text_r.squeeze(2)
            # print("Encoded phone ", enc_text_r.shape) # [b, featsize]
            # print(enc_audio_r.shape, enc_text_r.shape)

            ########################
            # print(trans.shape)
            enc_trans = self.wordEncode(trans, bert_len)
            enc_trans_r = enc_trans.transpose(2,1)
            enc_trans_r = nn.functional.avg_pool1d(enc_trans_r, [enc_trans_r.size()[2]], stride=1)
            enc_trans_r = enc_trans_r.squeeze(2)
            # print("Encoded phone ", enc_trans_r.shape) # [b, featsize]
            
            enc = torch.cat((enc_audio_r, enc_text_r, enc_trans_r ), dim=1) #[b, featsize*2]
            encitems.append(enc)
            # print(enc.shape)
        encitems = torch.stack(encitems).cuda()
        # print(encitems.shape)
        enc = torch.mean(encitems, dim=0)
        # print(enc.shape)
        # enc = torch.cat((enc_audio_r, enc_text_r ), dim=1)

        # enc = torch.cat((enc_audio_r, enc_text_r), dim=1)
        logits = self.fclayer(enc)
        logits =  nn.functional.log_softmax(logits, dim=-1)
        # print("Logits: ", logits.shape)
        return enc, logits

    def forward2(self, src, txt, trans, wav_len=None, bert_len=None, pad_idx=0):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        pad_idx : int, optional
            The index for <pad> token (default=0).
        """

        # reshpae the src vector to [Batch, Time, Fea] is a 4d vector is given
        src = self.custom_src_module(src)
        txt = self.phoneEmb(txt.squeeze(2))
        txt = self.custom_src_module_phone(txt)
        trans = self.custom_src_module_bert(trans)
        # print(src.shape, txt.shape, trans.shape)
        # print(src.shape, txt.shape)
        src = torch.cat((src, txt, trans), dim=2)
        enc_audio = self.audioEncode2(src, wav_len)
        enc_audio_r = enc_audio.transpose(2,1)
        # print("Encoded output shape: ", enc_audio_r.shape)
        enc_audio_r = self.convmodule(enc_audio_r)
        # print("Encoded output after convmodule: ", enc_audio_r.shape)
       
        enc_audio_r = nn.functional.avg_pool1d(enc_audio_r, [enc_audio_r.size()[2]], stride=1)
        enc_audio_r = enc_audio_r.squeeze(2)

        logits = self.fclayer(enc_audio_r)
        logits =  nn.functional.log_softmax(logits, dim=-1)
        return enc_audio_r, logits


    def forward_unaligned(self, src, txt, trans, wav_len=None, bert_len=None, pad_idx=0):
        """
        txt is the phoneme, trans is bert for words
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        pad_idx : int, optional
            The index for <pad> token (default=0).
        """

        # reshpae the src vector to [Batch, Time, Fea] is a 4d vector is given
        print("Audio: ", src.shape)
        enc_audio = self.audioEncode(src, wav_len) 
        print("1: ", enc_audio.shape)
        enc_audio_r = enc_audio.transpose(2,1)
        enc_audio_r = nn.functional.avg_pool1d(enc_audio_r, [enc_audio_r.size()[2]], stride=1)
        enc_audio_r = enc_audio_r.squeeze(2)
        # print("Encoded Audio: ", enc_audio_r.shape) # [b, featsize]
        # print("Txt shape", txt.shape) #[b, t', 1]
        print("Txt shape: ", txt.shape)
        txt = self.phoneEmb(txt.squeeze(2))
        # print(txt.shape)
        enc_text = self.phoneEncode(txt, wav_len)
        enc_text_r = enc_text.transpose(2,1)
        enc_text_r = nn.functional.avg_pool1d(enc_text_r, [enc_text_r.size()[2]], stride=1)
        enc_text_r = enc_text_r.squeeze(2)

        ######
        # print(txt.shape)
        print("Trans shape: ", trans.shape)
        enc_trans = self.wordEncode(trans, bert_len)
        enc_trans_r = enc_trans.transpose(2,1)
        enc_trans_r = nn.functional.avg_pool1d(enc_trans_r, [enc_trans_r.size()[2]], stride=1)
        enc_trans_r = enc_trans_r.squeeze(2)

        # print("Encoded phone ", enc_text_r.shape) # [b, featsize]
        enc = torch.cat((enc_audio_r, enc_text_r, enc_trans_r ), dim=1) #[b, featsize*2]
        logits = self.fclayer(enc)
        logits =  nn.functional.log_softmax(logits, dim=-1)
        # import pdb 
        # pdb.set_trace()
        return enc, logits

    def forward_unaligned_2(self, src, txt, trans, wav_len=None, bert_len=None, pad_idx=0):
        """
        txt is the phoneme, trans is bert for words
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        pad_idx : int, optional
            The index for <pad> token (default=0).
        """

        # reshpae the src vector to [Batch, Time, Fea] is a 4d vector is given
        enc_audio = self.audioEncode(src, wav_len) 
        # print("1: ", enc_audio.shape)
        enc_audio_r = enc_audio.transpose(2,1)
        enc_audio_r = nn.functional.avg_pool1d(enc_audio_r, [enc_audio_r.size()[2]], stride=1)
        enc_audio_r = enc_audio_r.squeeze(2)
        # print("Encoded Audio: ", enc_audio_r.shape) # [b, featsize]
        # print("Txt shape", txt.shape) #[b, t', 1]

        txt = self.phoneEmb(txt.squeeze(2))
        # print(txt.shape)
        enc_text = self.phoneEncode(txt, wav_len)
        enc_text_r = enc_text.transpose(2,1)
        enc_text_r = nn.functional.avg_pool1d(enc_text_r, [enc_text_r.size()[2]], stride=1)
        enc_text_r = enc_text_r.squeeze(2)

        ######
        # print(txt.shape)
        enc_trans = self.wordEncode(trans, bert_len)
        enc_trans_r = enc_trans.transpose(2,1)
        enc_trans_r = nn.functional.avg_pool1d(enc_trans_r, [enc_trans_r.size()[2]], stride=1)
        enc_trans_r = enc_trans_r.squeeze(2)

        # print("Encoded phone ", enc_text_r.shape) # [b, featsize]
        enc = torch.cat((enc_audio_r, enc_text_r, enc_trans_r ), dim=1) #[b, featsize*2]
        logits = self.fclayer(enc)
        logits =  nn.functional.log_softmax(logits, dim=-1)
        # import pdb 
        # pdb.set_trace()
        return enc, logits

    def forward_aligned_unaligned(self, src, txt_aligned, txt_unaligned,\
                    trans_aligned , trans_unaligned, \
                    wav_len, trans_aligned_len, trans_unaligned_len , \
                    phone_aligned_len, phone_unaligned_len):

        enc_audio = self.audioEncode(src, wav_len)
        # print(enc_audio.shape)
        enc_audio_r = enc_audio.transpose(2,1)
        enc_audio_r = nn.functional.avg_pool1d(enc_audio_r, [enc_audio_r.size()[2]], stride=1)
        enc_audio_r = enc_audio_r.squeeze(2)
        # print("Encoded Audio: ", enc_audio_r.shape)
        # print("Txt shape", txt.shape)

        txt = self.phoneEmb(txt_aligned.squeeze(2))
        # print(txt.shape)
        enc_text = self.phoneEncode(txt, phone_aligned_len)
        enc_text_r = enc_text.transpose(2,1)
        enc_text_r = nn.functional.avg_pool1d(enc_text_r, [enc_text_r.size()[2]], stride=1)
        enc_text_r = enc_text_r.squeeze(2)
        # print(enc_audio_r.shape, enc_text_r.shape)

        ########################
        # print(trans.shape)
        enc_trans = self.wordEncode(trans_aligned, trans_aligned_len)
        enc_trans_r = enc_trans.transpose(2,1)
        enc_trans_r = nn.functional.avg_pool1d(enc_trans_r, [enc_trans_r.size()[2]], stride=1)
        enc_trans_r = enc_trans_r.squeeze(2)

        ################################
        txt_unaligned = self.phoneEmb(txt_unaligned.squeeze(2))
        enc_text_un = self.phoneEncode(txt_unaligned, phone_unaligned_len)
        enc_text_un_r = enc_text_un.transpose(2,1)
        enc_text_un_r = nn.functional.avg_pool1d(enc_text_un_r, [enc_text_un_r.size()[2]], stride=1)
        enc_text_un_r = enc_text_un_r.squeeze(2)

        ########################
        # print(trans.shape)
        enc_trans_un = self.wordEncode(trans_unaligned, trans_unaligned_len)
        enc_trans_un_r = enc_trans_un.transpose(2,1)
        enc_trans_un_r = nn.functional.avg_pool1d(enc_trans_un_r, [enc_trans_un_r.size()[2]], stride=1)
        enc_trans_un_r = enc_trans_un_r.squeeze(2)

        #########################

        # print(enc_audio_r.shape, enc_text_r.shape, enc_trans_r.shape, enc_text_un_r.shape, enc_trans_un_r.shape)

        enc = torch.cat((enc_audio_r, enc_text_r, enc_trans_r, enc_text_un_r, enc_trans_un_r ), dim=1) #[b, featsize*2]
        # print(enc.shape )
        # enc = torch.cat((enc_audio_r, enc_text_r, enc_text_un_r ), dim=1)


        # enc = torch.cat((enc_audio_r, enc_text_r), dim=1)
        logits = self.fclayer(enc)
        logits =  nn.functional.log_softmax(logits, dim=-1)

        return enc, logits

    def forward_aligned_unaligned2(self, src, txt_aligned, txt_unaligned,\
                    trans_aligned , trans_unaligned, \
                    wav_len, trans_aligned_len, trans_unaligned_len , \
                    phone_aligned_len, phone_unaligned_len):

        src = self.custom_src_module(src)
        txt = self.phoneEmb(txt_aligned.squeeze(2))
        txt = self.custom_src_module_phone(txt)
        trans = self.custom_src_module_bert(trans_aligned)
        # print(src.shape, txt.shape, trans.shape)
        src = torch.cat((src, txt, trans), dim=2)
        # print(src.shape)
        enc_audio = self.audioEncode2(src, wav_len)
        enc_audio_r = enc_audio.transpose(2,1)
        enc_audio_r = nn.functional.avg_pool1d(enc_audio_r, [enc_audio_r.size()[2]], stride=1)
        enc_audio_r = enc_audio_r.squeeze(2)

        ################################
        txt_unaligned = self.phoneEmb(txt_unaligned.squeeze(2))
        # print(txt_unaligned.shape)
        enc_text_un = self.phoneEncode(txt_unaligned, phone_unaligned_len)
        enc_text_un_r = enc_text_un.transpose(2,1)
        enc_text_un_r = nn.functional.avg_pool1d(enc_text_un_r, [enc_text_un_r.size()[2]], stride=1)
        enc_text_un_r = enc_text_un_r.squeeze(2)

        ########################
        # print(trans.shape)
        enc_trans_un = self.wordEncode(trans_unaligned, trans_unaligned_len)
        enc_trans_un_r = enc_trans_un.transpose(2,1)
        enc_trans_un_r = nn.functional.avg_pool1d(enc_trans_un_r, [enc_trans_un_r.size()[2]], stride=1)
        enc_trans_un_r = enc_trans_un_r.squeeze(2)

        # print(enc_audio_r.shape, enc_text_un_r.shape)#, enc_trans_un_r.shape)

        enc = torch.cat((enc_audio_r, enc_text_un_r , enc_trans_un_r), dim=1) #[b, featsize*2]
        # print(enc.shape)

        logits = self.fclayer(enc)
        logits =  nn.functional.log_softmax(logits, dim=-1)
        return enc, logits


    def make_masks(self, src, tgt, wav_len=None, pad_idx=0):
        """This method generates the masks for training the transformer model.

        Arguments
        ---------
        src : tensor
            The sequence to the encoder (required).
        tgt : tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).
        """
        src_key_padding_mask = None
        if wav_len is not None:
            abs_len = torch.round(wav_len * src.shape[1])
            src_key_padding_mask = (
                torch.arange(src.shape[1])[None, :].to(abs_len)
                > abs_len[:, None]
            )

        tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)

        src_mask = None
        tgt_mask = get_lookahead_mask(tgt)
        return src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask

    def audioEncode2(
        self, src, wav_len=None,
    ):
        """
        Encoder forward pass

        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        """
        # reshape the src vector to [Batch, Time, Fea] if a 4d vector is given
        if src.dim() == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        src_key_padding_mask = None
        if wav_len is not None:
            abs_len = torch.floor(wav_len * src.shape[1])
            src_key_padding_mask = (
                torch.arange(src.shape[1])[None, :].to(abs_len)
                > abs_len[:, None]
            )

        # src = self.custom_src_module(src) # linear and dropout
        if self.attention_type == "RelPosMHAXL":
            pos_embs_source = self.positional_encoding(src)

        elif self.positional_encoding_type == "fixed_abs_sine":
            src = src + self.positional_encoding(src)
            # print(self.positional_encoding)
            # print("positional encoding: ", self.positional_encoding(src))
            pos_embs_source = None

        encoder_out, attention_lst = self.encoder(
            src=src,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_source,
        )
        return encoder_out, attention_lst

    def audioEncode(
        self, src, wav_len=None,
    ):
        """
        Encoder forward pass

        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        """
        # reshape the src vector to [Batch, Time, Fea] if a 4d vector is given
        if src.dim() == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        src_key_padding_mask = None
        if wav_len is not None:
            abs_len = torch.floor(wav_len * src.shape[1])
            src_key_padding_mask = (
                torch.arange(src.shape[1])[None, :].to(abs_len)
                > abs_len[:, None]
            )

        src = self.custom_src_module(src) # linear and dropout
        if self.attention_type == "RelPosMHAXL":
            pos_embs_source = self.positional_encoding(src)

        elif self.positional_encoding_type == "fixed_abs_sine":
            src = src + self.positional_encoding(src)
            # print(self.positional_encoding)
            # print("audioEncode positional encoding: ", self.positional_encoding(src).shape)
            pos_embs_source = None

        encoder_out, attention_lst = self.encoder(
            src=src,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_source,
        )
        return encoder_out, attention_lst

    def phoneEncode(
        self, src, wav_len=None,
    ):
        """
        Encoder forward pass

        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        """
        # reshape the src vector to [Batch, Time, Fea] if a 4d vector is given
        if src.dim() == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        src_key_padding_mask = None
        if wav_len is not None:
            abs_len = torch.floor(wav_len * src.shape[1])
            src_key_padding_mask = (
                torch.arange(src.shape[1])[None, :].to(abs_len)
                > abs_len[:, None]
            )

        src = self.custom_src_module_phone(src) # linear and dropout
        if self.attention_type == "RelPosMHAXL":
            pos_embs_source = self.positional_encoding(src)

        elif self.positional_encoding_type == "fixed_abs_sine":
            
            if self.modalities == "a_phfeat_aligned_unaligned" or self.modalities == "a_phfeat_aligned_unaligned-2":
                src = src + self.positional_encoding2(src)
            else:
                src = src + self.positional_encoding(src)
            # print(self.positional_encoding)
            # print("positional encoding: ", self.positional_encoding(src))
            pos_embs_source = None

        encoder_out, attention_lst = self.textencoder(
            src=src,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_source,
        )
        return encoder_out, attention_lst

    def wordEncode(
        self, src, wav_len=None,
    ):
        """
        Encoder forward pass

        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        """
        # reshape the src vector to [Batch, Time, Fea] if a 4d vector is given
        if src.dim() == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        src_key_padding_mask = None
        if wav_len is not None:
            abs_len = torch.floor(wav_len * src.shape[1])
            src_key_padding_mask = (
                torch.arange(src.shape[1])[None, :].to(abs_len)
                > abs_len[:, None]
            )
        # print(src.shape)
        src = self.custom_src_module_bert(src) # linear and dropout
        # print(src.shape)
        if self.attention_type == "RelPosMHAXL":
            pos_embs_source = self.positional_encoding(src)

        elif self.positional_encoding_type == "fixed_abs_sine":
            # print("Word encoding, ", src.shape, self.positional_encoding(src).shape)
          
            if self.modalities == "a_phfeat_aligned_unaligned" or self.modalities == "a_phfeat_aligned_unaligned-2":
                src = src + self.positional_encoding2(src)
            else:
                src = src + self.positional_encoding(src)
            # print(self.positional_encoding)

            # print("positional encoding: ", self.positional_encoding(src))
            pos_embs_source = None

        encoder_out, attention_lst = self.wordencoder(
            src=src,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_source,
        )
        return encoder_out, attention_lst

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)


if __name__ == "__main__":
    # src = torch.rand([8, 120, 512]) # batch, time, feature
    # tgt = torch.randint(0, 720, [8, 120])
    src = torch.rand([8, 80, 40])
    src_lens = torch.Tensor([20,18,17,16,16,15,15,14])

    net = TransformerClass(
        #tgt_vocab, input_size, d_model, nhead, num_encoder_layer, num_decoder_layer, d_ffn, n_class
        720, 40, 768, 8, 1, 1, 1024, 4, activation=torch.nn.GELU
    )
    # classifier = TransformerClassiferWrapper(net)
    out = net(src=src,wav_len=src_lens)
    # out = net.forward(src)
    print(out.shape) #torch.Size([8, 120, 512])