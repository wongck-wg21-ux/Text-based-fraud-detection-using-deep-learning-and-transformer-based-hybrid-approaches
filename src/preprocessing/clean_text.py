import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Extended disfluency list for call transcripts
disfluencies = [
    # Fillers and Hesitations
    "uh", "um", "er", "ah", "hmm", "huh", "oh", "mmm",
    
    # Hedges and Conversational Fillers
    "like", "you know", "i mean", "sort of", "kind of", "actually", "basically",
    
    # Backchannels and Turn-Taking Cues
    "uh-huh", "alright", "okay", "well", "right", "so", "yeah", "yep", "nope"
]

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Input must be text")

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)

    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(w) for w in tokens
        if w not in stop_words and w not in disfluencies
    ]

    return " ".join(tokens)
