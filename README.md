## Recipes for Analyzing Infant/Parent Vocalization and Children's Speech 
These recipes are developed based on an early version (0.5.3) of [SpeechBrain toolkit](https://github.com/speechbrain/speechbrain). 

### Uses
To check out this repository and install speechbrain,
```
git clone https://github.com/jialuli3/wav2vec_LittleBeats_LENA.git
cd wav2vec_LittleBeats_LENA
cd speechbrain
pip install -r requirements.txt
pip install --editable .
```

### Directory for each recipe
This recipe contains scripts for 
- Training parent/infant speaker diarization and vocalization classifications on LittleBeats home recordings. 

  ```
  cd recipes/wav2vec_LittleBeats
  ```
- perform children's phoneme recognition
  ```
  cd recipes/LibriSpeech/ASR/CTC_Children_PR
  ```
- perform vocalization classification in BabbleCor with Phoneme Recognition features 
  ```
  cd recipes/BabbleCor
  ```
**Readme.md document is available under each directory for more detailed walkthrough.**

### Paper/BibTex Citation
If you found this recipe or our paper helpful, please cite at least one of our references as
```
@inproceedings{li23e_interspeech,
  author={Jialu Li and Mark Hasegawa-Johnson and Nancy L. McElwain},
  title={{Towards Robust Family-Infant Audio Analysis Based on Unsupervised Pretraining of Wav2vec 2.0 on Large-Scale Unlabeled Family Audio}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={1035--1039},
  doi={10.21437/Interspeech.2023-460}
}
```
```
@INPROCEEDINGS{10626416,
  author={Li, Jialu and Hasegawa-Johnson, Mark and McElwain, Nancy L.},
  booktitle={2024 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW)}, 
  title={Analysis of Self-Supervised Speech Models on Children’s Speech and Infant Vocalizations}, 
  year={2024},
  pages={550-554},
  doi={10.1109/ICASSPW62465.2024.10626416}}

```
```
@inproceedings{li24j_interspeech,
  title     = {Enhancing Child Vocalization Classification with  Phonetically-Tuned Embeddings for Assisting Autism Diagnosis},
  author    = {Jialu Li and Mark Hasegawa-Johnson and Karrie Karahalios},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {5163--5167},
  doi       = {10.21437/Interspeech.2024-540},
  issn      = {2958-1796},
}
```
### Contact
Jialu Li (she, her, hers)

E-mail: jialuli3@illinois.edu

Homepage: https://sites.google.com/view/jialuli/

Our team: https://littlebeats.hdfs.illinois.edu/team/
