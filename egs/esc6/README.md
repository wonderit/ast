## Setting
Step 1. Clone or download this repository and set it as the working directory, create a virtual environment and install the dependencies.
```shell
cd ast/ 
conda create -n ast python=3.11
conda activate ast
pip install -r requirements-linux/mac.txt 
```
Step 2. Test the AST model.

```shell
cd ast/src
python 
```

```shell
import os 
import torch
from models import ASTModel 
# download pretrained model in this directory
os.environ['TORCH_HOME'] = '../pretrained_models'  
# assume each input spectrogram has 100 time frames
input_tdim = 100
# assume the task has 527 classes
label_dim = 527
# create a pseudo input: a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins 
test_input = torch.rand([10, input_tdim, 128]) 
# create an AST model
ast_mdl = ASTModel(label_dim=label_dim, input_tdim=input_tdim, imagenet_pretrain=True)
test_output = ast_mdl(test_input) 
# output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes. 
print(test_output.shape)  
```

Step 3. add esc-6 dataset
```text
egs
├── esc6
│   └── data                
│        ├── audio-one-dir  # add this dataset folder
│        └── datafiles      # add 5-fold dataset scheme
└── ...

```
esc6_train/eval_data_k.json files are generated from `data/datafiles/prep_esc6.py`

## Train ESC-6 Recipe
```shell
cd ast/egs/esc6
(local user) ./run_esc6.sh
```

Result Summary
```text
--------------Result Summary--------------
Fold 0 accuracy: 0.9167
Fold 1 accuracy: 1.0000
Fold 2 accuracy: 0.9375
Fold 3 accuracy: 0.9583
Fold 4 accuracy: 1.0000
The averaged accuracy of 5 folds is 0.963
```

## Inference ESC-6 Recipe
check [AST_Inference_Demo_esc6.ipynb](AST_Inference_Demo_esc6.ipynb)

result for `'./data/audio-one-dir/1-17124-A.wav'`
```text
Predice results:
- Car_horn: 0.9827
- Engine: 0.3404
- Fireworks: 0.3268
- Crying_baby: 0.2313
- Siren: 0.2163
- Clapping: 0.1540
```

### c.f.) for audio_length
This makes sense. But you also need to take care of the hyper-parameters, in particular, audio_length should be the max length of frames of audios in your dataset (e.g., 100 for 1s audio) timem should be about 20% of your average audio length, e.g., 25 for 1s audio. You also need to tune learning rate, etc.
