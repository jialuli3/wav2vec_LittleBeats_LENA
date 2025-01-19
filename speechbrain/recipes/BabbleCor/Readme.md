## Recipe for children's vocalization classification with features from phoneme Recognition of BabbleCor   
This recipe is developed based on SpeechBrain toolkit. This recipe contains scripts for classifying BabbleCor used in the [Baby Sound Sub-Challenge in 2019 InterSpeech Paralinguistic Chanllenges](https://www.isca-archive.org/interspeech_2019/schuller19_interspeech.pdf) with wav2vec 2.0 models.

### About BabbleCor
[BabbleCor](https://pubmed.ncbi.nlm.nih.gov/33497512/) contains 11k short audio clips of 52 healthy children aged 2-36 months old. Five types of vocalizations are classified, including 
- canonical babbling (containing a consonant to vowel transition)
- non-canonical babbling (not containing a consonant to vowel transition)
- laughing
- crying
- junk

## Uses
### Install SpeechBrain
```
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
pip install -r requirements.txt
pip install --editable .

```

### Check out this branch
```
git clone https://github.com/jialuli3/speechbrain.git
cd speechbrain
git checkout -b infant-voc-classification
git pull origin infant-voc-classification
```

### Change to BabbleCor directory
```
cd recipes/BabbleCor
```

### Download wav2vec2 model fine-tuned with children's phoneme recognition task on Providence corpus

https://huggingface.co/lijialudew/wav2vec_children_ASR


### Prepare data in json format ###
To make data compatible with this script, prepare your data similar as the following json format
```
{
  "sample_data1": {
    "wav": "path/to/your/wav/file",
    "label": "one of the babblecor label type",
    "child_ID": childen's ID,
    "hyp_pr": hypothesis transcript generated from the phoneme recognition model # this is necessary for inferencing the phoneme recognition transcript as auxiliary task
    }
}
```

Sample json file we used in our experiments can be found in **sample_json/sample_babblecor.json**

### Make yaml files in *hparams* folder compatiable with your dataset
- Change data paths in *train_annotation*, *valid_annotation*, and *test_annotation*
- To train with additional phoneme recognition features, copy and paste *wav2vec2_asr.ckpt* from pretrained wav2vec checkpoints fine-tuned with children's phoneme recognition to *save_folder* directory.

### Fine-tune wav2vec2 model with BabbleCor ###

Run the following commands to fine-tune wav2vec2 using our developed recipe

```
# Train wav2vec2 without phoneme recognition features
python train_1_w2v2_WA_2dnn.py hparams/train_1_w2v2_2dnn_WA_LL4300_bbcor.yaml

# Train wav2vec2 with phoneme recognition features
python train_1_w2v2_WA_2dnn_combine_asr_features_bbcor.py hparams/train_1_w2v2_2dnn_WA_LL4300_asr_bbcor_concat.yaml
```

### Paper/BibTex Citation
If you found this recipe or our paper helpful, please cite us as

```
@article{li2023enhancing,
  title={Enhancing Child Vocalization Classification with Phonetically-Tuned Embeddings for Assisting Autism Diagnosis},
  author={Li, Jialu and Hasegawa-Johnson, Mark and Karahalios, Karrie},
  booktitle={Interspeech},
  year={2024}
}
```

### Contact
Jialu Li (she, her, hers)

E-mail: jialuli3@illinois.edu

Homepage: https://sites.google.com/view/jialuli/