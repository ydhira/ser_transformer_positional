# ser_transformer_positional

## Description:
This repository contains the codebase used for the publication of the paper `Positional Encoding for Capturing Modality Specific Cadence for Emotion Detection`. 
We implement the Speech Emotion Recognition (SER) task and for each input instance, we use the audio signal, phoneme sequence and the word sequence.
The main idea proposed in the paper suggests that the three input streams are even though related to each other, also have their individual local `cadence`, 
which is important to be modelled for the task of SER. We model the local cadence by `positional encodings` - that are a part of the transformer architecture. 
We model individual cadences by having separate positional encodings for each input stream. 
Our results show that emotion detection based on this strategy is better than when the
modality specific cadence is ignored or normalized out by using a shared positional encoding. We also find that capturing
the modality interdependence is not as important as is capturing of the local cadence of individual modalities. We conduct
our experiments on the `IEMOCAP` and `CMU-MOSI` datasets to
demonstrate the effectiveness of the proposed methodology for
combining multi-modal evidence.

## Find our paper here: 
https://drive.google.com/file/d/1RbUv7dmYecJUhJxqaofwNrAZPv5bvXzC/view

## Cite our work:
@article{dhamyal2022positional,
  title={Positional Encoding for Capturing Modality Specific Cadence for Emotion Detection},
  author={Dhamyal, Hira and Raj, Bhiksha and Singh, Rita},
  journal={Proc. Interspeech 2022},
  pages={166--170},
  year={2022}
}
