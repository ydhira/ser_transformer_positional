from transformers import Wav2Vec2Processor, HubertModel
# from datasets import load_dataset
import soundfile as sf

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")


# def map_to_array(batch):
#     speech, _ = sf.read(batch["file"])
#     batch["speech"] = speech
#     return batch


# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# ds = ds.map(map_to_array)
audio_file='/home/hyd/workhorse2/multi-hop-attention/data/IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav'

audio = sf.read(audio_file)[0]
audio1 = sf.read('/home/hyd/workhorse2/multi-hop-attention/data/IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F001.wav')[0]
# print(audio)
input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values  # Batch size 1
# print(input_values.shape)
hidden_states = model(input_values).last_hidden_state

print(hidden_states.squeeze().detach().numpy().shape)
h=hidden_states.squeeze().detach().numpy()
print(type(hidden_states))
# audio = sf.read(audio_file)[0]
# input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values  # Batch size 1
# # print(input_values.shape)
# hidden_states = model(input_values).last_hidden_state
# print(hidden_states.shape)