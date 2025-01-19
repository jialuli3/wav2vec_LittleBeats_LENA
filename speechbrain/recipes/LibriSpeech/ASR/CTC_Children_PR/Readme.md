## Children's Phoneme Recognition Recipe 
This recipe is developed based on SpeechBrain toolkit. This recipe contains scripts for training children's phoneme recognition using wav2vec 2.0 model. 

This recipe provides a sample of training phoneme recognition on My Science Tutor (MyST). Similar recipe can be produced for training Providence corpus.

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

### Change to the directory of recipe
```
cd recipes/LibriSpeech/ASR/CTC_Children_PR
```

### Download corpora and preprocess text scripts
Download [MyST](https://catalog.ldc.upenn.edu/LDC2021S05) and [Providence](https://phon.talkbank.org/access/Eng-NA/Providence.html) corpora. Convert transcripts of MyST to IPA format. We use [eng_to_ipa](https://pypi.org/project/eng-to-ipa/) software. 

Install eng_to_ipa package with the following command line
```
pip install eng-to-ipa
```

### Prepare data in JSON file
To make data compatible with this script, prepare your data similar as the following json format. 
```
{
  "sample_data1": { 
    "wav": "path/to/your/wav/file1",
    "raw_txt": "transcript of wav file",
    "dur": 5.2 #duration of current wav file,
    "spk_id": "002116" # speaker ID,
    }
}
```

Then convert provided transcript into IPA format, see a sample preprocessing script in **preprocessing_scripts/preprocess_MyST.py**. The output JSON file should look like a sample JSON file in **sample_json/sample_MyST.json**.

For Providence corpus, directly map SAMPA symbols to IPA format based on the similarity of pronounciations. 

### Make yaml files in *hparams* folder compatiable with your dataset
- Change data path in *data_folder* to make *train_json*, *valid_json*, and *test_json* valid
- Get the pretrained model checkpoint, [*W2V2-Libri100*](https://huggingface.co/lijialudew/wav2vec_children_ASR/tree/main/save_100h), from LibriSpeech corpus from our [HuggingFace repo](https://huggingface.co/lijialudew/wav2vec_children_ASR), or pretrain from scratch using original speechbrain toolkit.
- Change *save_folder* to the path to the pretrained model checkpoint.

### Fine-tune wav2vec2 model with MyST corpus
Run the following commands to fine-tune wav2vec2 using our developed recipe

```
python train_with_wav2vec_phos_MyST_Providence.py hparams/train_with_wav2vec_phos_Libri_MyST_Providence.yaml
```
### Download all pretrained model weights ###

Our pretrained model weights can be downloaded via our Hugging Face repository

https://huggingface.co/lijialudew/wav2vec_children_ASR/tree/main

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
and/or
```
@inproceedings{li2024analysis,
  title={Analysis of Self-Supervised Speech Models on Children's Speech and Infant Vocalizations},
  author={Li, Jialu and Hasegawa-Johnson, Mark and McElwain, Nancy L},
  booktitle={IEEE Workshop on Self-Supervision in Audio, Speech and Beyond (SASB)},
  year={2024}
}
```

### Contact
Jialu Li (she, her, hers)

E-mail: jialuli3@illinois.edu

Homepage: https://sites.google.com/view/jialuli/