import re
import pronouncing
import nltk
from nltk.corpus import cmudict
import spacy

nltk.download('cmudict')  # used in if gh is silent 
cmudict_dict = cmudict.dict()


def g_is_silent(word): # check if 'g' is silent before 'n'
    if 'gn' not in word.lower():
        return False
    prons = cmudict_dict.get(word.lower())
    if not prons:
        return False

    for pron in prons:
        if any(p.startswith("G") for p in pron):
            return False  
    return True  

def is_gh_silent(word):
    if word.lower() not in cmudict_dict:
        return False
    if 'gh' not in word.lower():  # Early return if no 'gh'
        return False
    
    pronunciations = cmudict_dict[word.lower()]
    for pron in pronunciations:
        phonemes = ''.join(pron)
        if not ('G' in phonemes or 'F' in phonemes):
            return True
    return False

def has_silent_letters(word):
    word_lower = word.lower()
    regex_match = False
    if word.lower() not in cmudict_dict:
        return False
    if g_is_silent(word_lower):
        regex_match = True

    elif is_gh_silent(word_lower):
        regex_match = True
    else:
        # complex_vowel_pattern reference: Dyslexic and typical-reading children use vowel digraphs as perceptual units in reading
        silent_letters_pattern = re.compile(r"\bkn|\bwr|\bps|\bts|\bmn|mn\b|mb\b")
        
        regex_match = bool(silent_letters_pattern.search(word_lower))
    return regex_match