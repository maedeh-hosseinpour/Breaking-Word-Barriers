import re
import textstat
from wordfreq import word_frequency
import pandas as pd
from wordfreq import zipf_frequency
import spacy
import pyphen
import nltk
from nltk.corpus import cmudict
import silent_detection


nlp = spacy.load("en_core_web_sm") 

def is_named_entity(word):
    doc = nlp(word)
    return any(ent.text == word and ent.label_ in ["PERSON", "GPE", "ORG"] for ent in doc.ents)

def conditional_lower(word):
    if isinstance(word, str) and not is_named_entity(word):
        return word.lower()
    return word

#Preprocessing the text
def clean_text(text)-> str:
    """Basic text cleaning for statistical analysis""" 
    
    # Replace common contractions
    text = text.replace("'s", " is")
    # Standardize sentence-ending punctuation spacing
    text = re.sub(r'([.,!?])', r'\1 ', text)
    # Fix multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def detect_difficult_words(text)-> dict:
    complex_vowel_pattern = re.compile(r'ea|ou|ei|ie|oe|ai|au|oo')
    # digraphs reference: Variability in the word-reading performance of dyslexic readers: Effects of letter length, phoneme length and digraph presence, Bigraph-syllable blending therapy in deep dyslexia
    difficult_digraphs_pattern = re.compile(r'th|ch|gh|sh|ph|wh')
    # homophones references: Surface Dyslexia in a Language Without Irregularly Spelled Words, Do dyslexics misread a ROWS for a ROSE?
    #http://www.singularis.ltd.uk/bifroest/misc/homophones-list.html
    homophones =set({
        "accessary","accessory","ad","add","ail","ale","air","heir","aisle","I'll","isle","all","awl","allowed","aloud","alms","arms","altar","alter","arc","ark","aren't","aunt","ate","eight","auger","augur","auk","orc","aural","oral","away","aweigh","awe","oar","or","ore","axel","axle","aye","eye","I","bail","bale","bait","bate","baize","bays","bald","bawled","ball","bawl","band","banned","bard","barred","bare","bear","bark","barque","baron","barren","base","bass","bay","bey","bazaar","bizarre","be","bee","beach","beech","bean","been","beat","beet","beau","bow","beer","bier","bel","bell","belle","berry","bury","berth","birth","bight","bite","byte","billed","build","blew","blue","bloc","block","boar","bore","board","bored","boarder","border","bold","bowled","boos","booze","born","borne","bough","bow","boy","buoy",
        "brae","bray","braid","brayed","braise","brays","braze","brake","break","bread","bred","brews","bruise","bridal","bridle","broach","brooch","bur","burr","but","butt","buy","by","bye","buyer","byre","calendar","calender","call","caul","canvas","canvass","cast","caste","caster","castor","caw","core","corps","cede","seed","ceiling","sealing","cell","sell","censer","censor","sensor","cent","scent","sent","cereal","serial","cheap","cheep","check","cheque","choir","quire","chord","cord","cite","sight","site","clack","claque","clew","clue","close","cloze","coal","kohl","coarse","course","coign","coin","colonel","kernel","complacent","complaisant","complement","compliment","coo","coup","cops","copse","council","counsel","creak","creek","crews","cruise","cue","queue","curb","kerb","currant","current","cymbol","symbol",
        "days","daze","dear","deer","descent","dissent","desert","dessert","dew","due","die","dye","discreet","discrete","doe","doh","dough","done","dun","douse","dowse","draft","draught","dual","duel","earn","urn","eery","eyrie","ewe","yew","you", "fair","fare","fate","fête","faun","fawn","fay","fey","faze","phase","feat","feet","few","phew","fie","phi","find","fined","fir","fur","flair","flare","flea","flee","flew","flu","flue","floe","flow","flour","flower","for","fore","four","gait","gate","genes","jeans","gild","guild","gilt","guilt","gneiss","nice","grate","great","groan","grown","guessed","guest","hail","hale","hair","hare","hall","haul","hangar","hanger","hart","heart","hay","hey","heal","heel","he'll","hear","here","heard","herd","he'd","heed","hew","hue","hi","high","higher","hire","him","hymn","ho","hoe","hoard","horde","hoarse","horse","hour","our",
        "idle","idol","in","inn","it's","its", "knead","need","knew","new","knight","night","knot","not","know","no","mail","male","main","mane","maize","maze","mare","mayor","meat","meet","mete","might","mite","moor","more","morning","mourning","muscle","mussel","naval","navel","nay","neigh","none","nun","one","won","pail","pale","pain","pane","pair","pare","pear","pea","pee","peace","piece","plain","plane","principal","principle","profit","prophet","rain","reign","rein","raise","rays","raze","read","reed","read","red","real","reel","right","rite","wright","write","road","rode","role","roll","root","route","rose","rows","sail","sale","scene","seen","sea","see","seam","seem","son","sun","some","sum","stair","stare","steal","steel","tail","tale","team","teem","there","their","they're","to","too","two","vain","vane","vein","wait","weight","war","wore","ware","wear","where","weak","week","weather","whether","which","witch","who's","whose","wood","would",
        "yoke","yolk"}) 

    words = set(word for word in re.findall(r'\b[a-zA-Z]+\b', text))  # Extract words
    words = {conditional_lower(word) for word in words}  # Convert to lowercase except named entities
    dic = pyphen.Pyphen(lang='en')
    # long words references: 1- Helping Students With Dyslexia Read Long Words: Using Syllables and Morphemes
    #2- The word-length effect in reading: A review

    longWords = []
    diagraphs=[]
    notFreqWords=[]
    silentLetters=[]
    VowelDigraphs=[]
    hasHomophones=[]
    difficultOrthography=[]
    hasComplexConsonants=[]

    for word in words:
        if zipf_frequency( word , 'en') <= 3:  
            notFreqWords.append(word)

        if len(word) >= 7: 
            longWords.append(word)

         # Silent letters with regex and is_gh_silent for gh cases and pronouncing library
        if silent_detection.has_silent_letters(word):
            silentLetters.append(word)

        if complex_vowel_pattern.search(word): # ou/ea/ei/ie/oe/ai/au/oo
            VowelDigraphs.append(word)

        if word.lower() in homophones:  
            #score += 1
            hasHomophones.append(word)

         # https://www.sciencedirect.com/science/article/pii/S0093934X96900553
         # https://www.sciencedirect.com/science/article/pii/S0022096517300784
        if sum(word.count(letter) for letter in "pqdb") / len(word) >= 0.4:
            difficultOrthography.append(word)

        if re.search(r'[^aeiou]{4,}', word):  # Four or more consonants in a row
            hasComplexConsonants.append(word)
        if difficult_digraphs_pattern.search(word):  # th/ch/gh/sh/ph/wh
            diagraphs.append(word)
        

    words_list = {
        "word_difficulty_lists": {
        "long_words": longWords,
        "diagraphs": diagraphs,
        "not_freq_words": notFreqWords,
        "silent_letters": silentLetters,
        "vowel_digraphs": VowelDigraphs,
        "has_homophones": hasHomophones,
        "difficult_orthography": difficultOrthography
                                }
                }
        
        
        #dys_toml.dict_toml_write(toml_data, file_path)

    return words_list

def get_text_stats(text) -> dict:
    """Get statistics for a text"""
    stats = { 
        'word_count': textstat.lexicon_count(text, removepunct=True),
        'sentence_count': textstat.sentence_count(text),
        'flesch_reading_ease': textstat.flesch_reading_ease(text), #Lower scores (0–30) indicate difficult texts (college level)
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text), #Higher values indicate more complex text.
        'difficult_words': textstat.difficult_words(text), #Words with more syllables and less frequent words
        'text_standard': textstat.text_standard(text), #combines multiple readability tests (like the Flesch Reading Ease, Flesch-Kincaid Grade Level, Dale-Chall, and others)
        'text_reading_time_estimate':textstat.reading_time(text, ms_per_char=14.69),
        'monosyllable_words_number':textstat.monosyllabcount(text)
            }
        
    return stats

#this function is used to compare the original text with the one generated by the model
def compare_texts(text1, text2)-> dict:
    """Compare statistics between two texts"""
    stats1 = get_text_stats(text1)
    stats2 = get_text_stats(text2)

    diff = {}
    for k in stats1:
        if isinstance(stats1[k], (int, float)) and isinstance(stats2[k], (int, float)):
            diff[k] = stats1[k] - stats2[k]
        else:
            diff[k] = "N/A"  # or leave out if you prefer

    df = pd.DataFrame({
        'original_text': stats1,
        'simplified_text': stats2,
        'difference': diff
    })

    df_dict = df.to_dict(orient="index")
    # dys_toml.dict_toml_write(df_dict, file_path)

    return df_dict

