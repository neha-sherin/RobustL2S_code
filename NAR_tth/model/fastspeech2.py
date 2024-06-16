import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

ConvTranspose2d = nn.ConvTranspose2d

from transformer import Encoder, Decoder, PostNet, HubertTransformer
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths,get_masked_with_pad_tensor,attention_image_summary
# import torch.nn.Transformer as Transformer
# from .layers import Decoder


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.inhubert_fc = nn.Linear(768, model_config["transformer"]["decoder_hidden"])
        self.relu = nn.ReLU()
        self.encoder = Encoder(model_config)

        #self.spectrogram_upsampler = SpectrogramUpsampler()
        self.hubconv = ConvTranspose2d(1, 1, [4,3], stride=[2,1], padding=[1,1]) #[1, 8])

        # self.decoder = Decoder(num_layers=model_config["transformer"]["decoder_layer"],
        #                 d_model=model_config["transformer"]["decoder_hidden"],
        #                 input_vocab_size=model_config["transformer"]["hubert_codes"]+3,
        #                 max_len=model_config["max_seq_len"])
        self.decoder = Decoder(model_config)
        #self.mel_linear = nn.Linear(model_config["transformer"]["decoder_hidden"], model_config["transformer"]["hubert_codes"],)

        # commented below two for parameter reduction. not needed for NAR
        #self.postnet = PostNet(model_config["transformer"]["hubert_codes"])
        #self.hubert_emb = nn.Embedding(model_config["transformer"]["hubert_codes"]+3,model_config["transformer"]["encoder_hidden"])
        self.hubert_fc = nn.Linear(model_config["transformer"]["decoder_hidden"],768)
        #self.START_TOKEN = model_config["transformer"]["hubert_codes"]+1
        #self.END_TOKEN = model_config["transformer"]["hubert_codes"]+2
        self.infer = False
        self.dropout = nn.Dropout(model_config["transformer"]["decoder_dropout"])


    def forward(
        self,
        #speakers,
        ids,
        texts,
        src_lens,
        max_src_len,
        huberts = None,
        huberts_lens = None,
        max_trg_len = None,
        d_targets = None,
        d_control=1.0
    ):
        # if self.infer == True:
        #     return self.inference(texts,src_lens,max_src_len,huberts,huberts_lens,max_trg_len)
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        #print(src_masks.shape)

        trg_masks = (
            get_mask_from_lengths(huberts_lens, max_trg_len)
            if huberts_lens is not None
            else None
        )

        # new_trg_masks = trg_masks==False
        # new_trg_masks = new_trg_masks
        # _,_,look_ahead_mask = get_masked_with_pad_tensor(max_trg_len,new_trg_masks,new_trg_masks,0)#get_mask_from_lengths(huberts_lens,max_trg_len)


        texts = self.hubconv(texts.unsqueeze(1))
        texts = F.relu(texts.squeeze(1))
        texts = self.inhubert_fc(texts)
        texts = self.relu(texts)
        #texts = self.spectrogram_upsampler(texts)
        encoder_out = self.encoder(texts, src_masks)

        # NAM
        # print((src_masks == trg_masks).all())
        mel_masks = src_masks
        output = encoder_out

        output, mel_masks = self.decoder(output, mel_masks)
        #output = self.mel_linear(output)

        # postnet_output = self.postnet(output) + output
        # start = torch.zeros_like(huberts[:,0]).unsqueeze(1).to(huberts.device)+self.START_TOKEN
        # huberts = torch.cat([start,huberts],dim=-1)[:,:-1]
        # output,W = self.decoder(huberts,encode_out=encoder_out,lookup_mask=look_ahead_mask)
        
        output = self.hubert_fc(output)
        output = self.dropout(output)
        
        if self.infer == True:
            #output = torch.argmax(output,dim=-1)
            output = output.squeeze(0)
            #print(output.shape)
            return output  #.tolist()

    
        return (
                output,
                src_masks,
                trg_masks,
                # postnet_output,
                # p_predictions,
                # e_predictions,
                # log_d_predictions,
                # d_rounded,
                # src_masks,
                # mel_masks,
                # src_lens,
                # mel_lens,
            )




    def test(self):
        self.eval()
        self.infer = True

    def inference(self,texts,src_lens,max_src_len,huberts = None,
    huberts_lens = None,
    max_trg_len = None):

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        encoder_out = self.encoder(texts, src_masks)
        # print("encoder out",encoder_out.shape)
        decoded = None
        # decoded = huberts.tolist()[:10]
        while True:
            # print('+++++++++++++')
            if decoded is None:
                decoded = [self.START_TOKEN]
            decoded_tensor = torch.tensor(decoded).to(encoder_out.device).unsqueeze(0)
            _,_,look_ahead_mask = get_masked_with_pad_tensor(len(decoded),decoded_tensor,decoded_tensor,0)#get_mask_from_lengths(huberts_lens,max_trg_len)
            # print(look_ahead_mask.shape)
            output,w = self.decoder(decoded_tensor,encode_out=encoder_out,lookup_mask=look_ahead_mask)
            output = self.hubert_fc(output)
            # else:
            #     decoded_tensor = torch.tensor(decoded).to(encoder_out.device).unsqueeze(0)
            #     output,w = self.decoder(decoded_tensor,encode_out=encoder_out)
            #     output = self.hubert_fc(output)
            current_hu = output[:,-1].argmax().item()
            if current_hu == self.END_TOKEN or len(decoded)==1000:
                break
            decoded.append(current_hu)


        return decoded[1:],w[-1]
        # if self.infer == True:
        #     return self.inference(texts,src_lens,max_src_len)
        #
        # src_masks = get_mask_from_lengths(src_lens, max_src_len)
        # trg_masks = get_mask_from_lengths(huberts_lens, max_trg_len)
        # new_trg_masks = trg_masks==False
        # _,_,look_ahead_mask = get_masked_with_pad_tensor(max_trg_len,new_trg_masks,new_trg_masks,0)#get_mask_from_lengths(huberts_lens,max_trg_len)
        # # print(look_ahead_mask)
        # encoder_out = self.encoder(texts, src_masks)
        # a = torch.zeros_like(huberts[:,0]).unsqueeze(1).to(huberts.device)+100
        # huberts = torch.cat([a,huberts],dim=-1)[:,:-1]
        # # print(huberts.shape)
        # output,w = self.decoder(huberts,encode_out=encoder_out,lookup_mask=look_ahead_mask)
        # output = self.hubert_fc(output)
        # decoded = output.argmax(-1)
        # attention_image_summary("trail1_s.png",w[-1])
        # print(decoded)
        # return decoded.tolist()[0]


class SpectrogramUpsampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])

    def forward(self, x):
        breakpoint()
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x
