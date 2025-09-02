# src/utils/bert_utils.py

from transformers import BertTokenizer, BertModel

def get_tokenizer(name="bert-base-uncased"):
    #Loads a Hugging Face tokenizer.
    return BertTokenizer.from_pretrained(name)

def load_bert(name="bert-base-uncased"):
    #Loads a pre-trained BERT model compatible with PyTorch.
    return BertModel.from_pretrained(name)
