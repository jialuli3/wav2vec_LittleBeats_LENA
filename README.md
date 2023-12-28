## LittleBeats family audio analysis: parent/infant speaker diarizatio and vocalization classification tasks 
This recipe is developed based on SpeechBrain toolkit. This recipe contains scripts for training parent/infant speaker diarization and vocalization classifications on LittleBeats home recordings using wav2vec 2.0 model.

### About LittleBeats
LittleBeats is a new infant wearable multi-modal device that we developed, which simultaneously records audio, movement of the infant, as well as heart-rate variablity. We use wav2vec2 to advance LB audio pipeline such that it automatically provides reliable labels of speaker diarization and vocalization classifications for family members, including infants, parents, and siblings, at home.

For more details, check out **https://littlebeats.hdfs.illinois.edu/**

## Uses
### Install SpeechBrain
```
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
pip install -r requirements.txt
pip install --editable .

```

### Download pretrained wav2vec2 models on LittleBeats and LENA audio ###

Our pretrained model weights can be downloaded via our Hugging Face repository

https://huggingface.co/lijialudew/wav2vec_LittleBeats_LENA/tree/main

### Check out this branch
```
git clone https://github.com/jialuli3/speechbrain.git
cd speechbrain
git checkout -b infant-voc-classification
git pull origin infant-voc-classification
```

### Prepare data in json format ###
To make data compatible with this script, prepare your data similar as the following json format
```
{
  "sample_data1": { # silence interval
    "wav_voc": "path/to/your/wav/file1",
    "dur_voc": 2.0,
    "sp": "SIL",
    "chn": "N",
    "fan": "N",
    "man": "N",
    "domain_label": "LB"
    },
  "sample_data2": { # CHN is babbling
    "wav_voc": "path/to/your/wav/file2",
    "dur_voc": 2.0,
    "sp": "CHN",
    "chn": "BAB",
    "fan": "N",
    "man": "N",
    "domain_label": "LENA"
    }
}
```
Speakers types include:
- **CHN**: target child
- **FAN**: female adult
- **MAN**: male adult
- **CXN**: sibling/other child

Vocalization types include:
- **CHN**
    - *CRY*: cry
    - *FUS*: fuss
    - *BAB*: babble
- **MAN/FAN**
    - *CDS*: child-directed speech
    - *MAN/FAN*: adult-directed speech
    - *LAU*: laugh
    - *SNG*: singing/rhythimic

Sample json file we used in our experiments can be found in **sample_json/sample_json.json**

### Make yaml files in *hparams* folder compatiable with your dataset
- Change data paths in *train_annotation*, *valid_annotation*, and *test_annotation*
- Download pretrained wav2vec checkpoints from our [HuggingFace repo](https://huggingface.co/lijialudew/wav2vec_LittleBeats_LENA/tree/main) and change *pretrained_path* under *wav2vec2* section accordingly
- If training with data augmentation, set *rir_folder* to a designated place for storing noise datapoints (noise data will be automatically downloaded)
- Change *data_folder* if prepared json file specifies relative path pointers (absolute path doesn't require to use this argument); read more about dataloader [here](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.dataio.dataio.html#speechbrain.dataio.dataio.load_data_json)

### Fine-tune wav2vec2 model on speaker diarization and parent/infant vocalization classification tasks ###
Before running Python script, first run
```
cd recipes/wav2vec_LittleBeats
```

Run the following commands to fine-tune wav2vec2 using our developed recipe

```
# Train wav2vec2 with features of last transformer layer
python scripts/train_3dnn.py hparams/hparams_LL_4300.yaml

# Train wav2vec2 with features over all transformer layers
python scripts/train_WA_3dnn.py hparams/hparams_LL_4300.yaml

# Train wav2vec2 with data augmentation on speaker diarization tier
python scripts/train_WA_wav_aug_on_sp.py hparams/hparams_LL_4300_WA_aug_on_sp.yaml

# Train wav2vec2 with domain embeddings, ECAPA-TDNN speaker embeddings on vocalization classification tiers, and data augmentation  on speaker diarization tier
python scripts/train_WA_wav_aug_on_sp_spk_domain.py hparams/hparams_LL_4300_WA_aug_spk_domain.yaml
```

### Paper/BibTex Citation
If you found this recipe or our paper helpful, please cite us as

Coming soon

### Contact
Jialu Li (she, her, hers)

Ph.D candidate @ Department of Electrical and Computer Engineering, University of Illinois at Urbana-Champaign

E-mail: jialuli3@illinois.edu

Homepage: https://sites.google.com/view/jialuli/

Our team: https://littlebeats.hdfs.illinois.edu/team/
