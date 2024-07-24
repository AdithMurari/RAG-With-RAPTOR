from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import nltk

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def get_wordnet_pos(word):
    """Get the WordNet part of speech for lemmatization."""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def get_synonyms(word, pos, top_n=3):
    synonyms = set()
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
        if len(synonyms) >= top_n:
            break
    return list(synonyms)[:top_n]

def lemmatize_words(words):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
    return lemmatized_words

def expand_query(query):
    words = word_tokenize(query)
    pos_tags = pos_tag(words)
    expanded_query = set(words)
    
    for word, tag in pos_tags:
        wn_pos = get_wordnet_pos(word)
        if wn_pos == wordnet.NOUN:
            # Synonym expansion for nouns
            expanded_query.update(get_synonyms(word, wn_pos, top_n=3))
            
    
    # Lemmatization
    expanded_query = lemmatize_words(expanded_query)
    
    return ' '.join(expanded_query)