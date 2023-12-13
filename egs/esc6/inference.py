import sys

sys.path.append("../../")
import torch, torchaudio, timm
import numpy as np
from torch.cuda.amp import autocast
import csv
import IPython
from src.models import ASTModel


class ASTModelVis(ASTModel):
    def get_att_map(self, block, x):
        qkv = block.attn.qkv
        num_heads = block.attn.num_heads
        scale = block.attn.scale
        B, N, C = x.shape
        qkv = qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        return attn

    def forward_visualization(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        # save the attention map of each of 12 Transformer layer
        att_list = []
        for blk in self.v.blocks:
            cur_att = self.get_att_map(blk, x)
            att_list.append(cur_att)
            x = blk(x)
        return att_list


# Create a new class that inherits the original ASTModel class
def make_features(wav_name, mel_bins, target_length=1024):
    waveform, sr = torchaudio.load(wav_name)
    # assert sr == 16000, 'input audio sampling rate must be 16kHz'

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10)

    print('sr', sr, fbank.shape)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank


def load_label(label_csv):
    with open(label_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    return labels


# Assume each input spectrogram has 300 time frames
input_tdim = 300
checkpoint_path = './exp/tiny224-esc6-f10-t10-impTrue-aspFalse-b48-lr1e-4/fold0/models/best_audio_model.pth'
# checkpoint_path = '/content/ast/pretrained_models/audio_mdl.pth'
# now load the visualization model
ast_mdl = ASTModelVis(label_dim=6, input_tdim=input_tdim, imagenet_pretrain=True, audioset_pretrain=False,
                      input_fdim=128, model_size='tiny224')
#
# ast_mdl = ASTModelVis(label_dim=6, fstride=10, tstride=10, input_fdim=128,
#                                   input_tdim=300, imagenet_pretrain=True,
#                                   audioset_pretrain=False, model_size='tiny224')

print(f'[*INFO] load checkpoint: {checkpoint_path}')
checkpoint = torch.load(checkpoint_path, map_location='cpu')
audio_model = torch.nn.DataParallel(ast_mdl)
audio_model.load_state_dict(checkpoint)
audio_model = audio_model.to(torch.device("cpu"))
audio_model.eval()

# Load the AudioSet label set
label_csv = './data/esc6_class_labels_indices.csv'  # label and indices for audioset data
labels = load_label(label_csv)

sample_audio_path = './data/audio-one-dir/1-17124-A.wav'
# sample_audio_path = './data/audio-one-dir/1-115920-B.wav'

feats = make_features(sample_audio_path, mel_bins=128, target_length=input_tdim)           # shape(1024, 128)
print(feats.shape)
feats_data = feats.expand(1, input_tdim, 128)           # reshape the feature
feats_data = feats_data.to(torch.device("cpu"))
# do some masking of the input
print(feats_data.shape)
# feats_data[:, :512, :] = 0.

# Make the prediction
with torch.no_grad():
  with autocast():
    output = audio_model.forward(feats_data)
    output = torch.sigmoid(output)
result_output = output.data.cpu().numpy()[0]
sorted_indexes = np.argsort(result_output)[::-1]

# Print audio tagging top probabilities
print('Predice results:')
for k in range(6):
    print('- {}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]], result_output[sorted_indexes[k]]))
print('Listen to this sample: ')
IPython.display.Audio(sample_audio_path)