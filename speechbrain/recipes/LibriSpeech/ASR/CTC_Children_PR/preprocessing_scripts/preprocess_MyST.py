import re
import eng_to_ipa as ipa 
import json
import os
from scipy.io.wavfile import read

def read_file(file):
    f=open(file,"r")
    content=f.readlines()
    f.close()
    return content[0].strip("\n")

def write_json(content,out):
    f=open(out,"w")
    f.write(json.dumps(content, sort_keys=False, indent=2))
    f.close()

def load_json(path):
    f=open(path,"r")
    return json.load(f)

ipa_symbols = [
    #two sounds
    "dz","ts","tS","dZ","kp","gb","tʃ","dʒ", 
    "aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ", "ɪə", "eə", "ʊə", "ɔə", "ɑɪ", "ɑʊ", "ɒɪ", "ɒʊ",
    # Consonants
    "p", "b", "t", "d", "ʈ", "ɖ", "c", "ɟ", "k", "ɡ", "q", "ɢ", "ʔ","g",
    "m", "ɱ", "n", "ɳ", "ɲ", "ŋ", "ɴ", "l",
    "ʙ", "r", "ʀ",
    "ⱱ", "ɾ", "ɽ",
    "ɸ", "β", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "ʂ", "ʐ", "ç", "ʝ", "x", "ɣ", "χ", "ʁ", "ħ", "ʕ", "h", "ɦ",
    "ɬ", "ɮ",
    "ʋ", "ɹ", "ɻ", "j", "ɰ",
    "ɭ", "ʎ", "ʟ",
    "ʍ", "w", "ɥ",
    "ʜ", "ʢ",
    "ɕ", "ʑ",
    "ɺ",
    "ɧ",

    # Vowels
    "i", "y", "ɨ", "ʉ", "ɯ", "u", "ɪ", "ʏ", "ʊ",
    "e", "ø", "ɘ", "ə","ɵ", "ɤ", "o", "ɛ", "œ", "ɜ", "ɞ", "ʌ", "ɔ",
    "æ", "ɐ", "a", "ɶ", "ä", "ɑ", "ɒ"
]

ipa_pattern = re.compile('|'.join(re.escape(sym) for sym in ipa_symbols))


def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        corpus = file.read()
    return corpus

def convert_to_ipa(text):
    output_text = ipa.convert(text).replace("ʤ","dʒ").replace("ʧ","tʃ")
    output_text = ipa_pattern.findall(output_text)
    return output_text
    
#def process_filled_pauses(text):
#    filled_pauses = ["UM", "ER", "UH", "AH", "HMM"]
#    for fp in filled_pauses:
#        text = re.sub(fr'\b{fp}\b', '', text, flags=re.IGNORECASE)
#    return text

def process_non_speech_events(text):
   non_speech_events = ["<BREATH>", "<LAUGH>", "<COUGH>", "<NOISE>", "<SIDE_SPEECH>",
                        "<NO_SIGNAL>", "<SILENCE>", "<SNIFF>", "<ECHO>", "<DISCARD>"]
   for event in non_speech_events:
       text = text.replace(event, ' ')
   return text

def process_special_symbols(text):
    text= re.sub(r'\+', '', text)
    text= re.sub(r'-', '', text)
    text= re.sub(r'\*', '', text)
    return text

def process_non_speech_events(text):
    return re.sub(r'\<.*?\>', '', text)

def process_unintelligible(text):
    return re.sub(r'\(\*\)', '', text)

def process_text_starting_with_parentheses(text):
    return re.sub(r'\(.*?\)', '', text)

def process_text_starting_with_double_parentheses(text):
    return re.sub(r'\(\((.*?)\)\)', '', text)

def clean_text(text):
    text = process_special_symbols(text)
    text = process_non_speech_events(text)
    text = process_unintelligible(text)
    text = process_text_starting_with_double_parentheses(text)
    text = process_text_starting_with_parentheses(text)
    text = re.sub(' +', ' ', text)
    text = text.strip()
    return text

data_root = "/path/to/myst-v0.4.2/data/train"
curr_dict={}
for root, dirs, files in os.walk(data_root):
    for file in files:
        if file.endswith("lab"):
            print(f"File: {os.path.join(root, file)}")
            try:
                key = file.replace(".lab","")
                spk_id = key.split("_")[1]
                wav_file = file.replace(".lab",".wav")
                rate,data = read(os.path.join(root, wav_file))
                dur = len(data)/rate
                txt=read_file(os.path.join(root, file))
                curr_dict[key]={
                    "wav": os.path.join(root, wav_file),
                    "raw_txt": txt,
                    "dur": dur,
                    "spk_id": spk_id
                }
            except:
                print("bad file")
                continue
write_json(curr_dict,"train.json")

input_dict=load_json("train.json")
new_dict = {}
for key,entry in input_dict.items():
    text = clean_text(entry["raw_txt"])
    if text=="": continue
    entry["cleaned_txt"] = text
    phonemes = convert_to_ipa(text)
    entry["phonemes"] = " ".join(phonemes)
    new_dict[key]=entry
write_json(new_dict,"train_cleaned.json")