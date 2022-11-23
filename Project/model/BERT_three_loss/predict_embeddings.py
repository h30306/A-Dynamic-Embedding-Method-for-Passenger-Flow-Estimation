from transformers import BertTokenizer
from BertForMaskNode import BertForMaskNode  
import torch
from IPython import embed

max_length = 13
embedding_size = 768

def BERT_vector(input_list, checkpoint_path):
    tk = BertTokenizer.from_pretrained(checkpoint_path)
    model = BertForMaskNode.from_pretrained(checkpoint_path)
    model = model.bert
    model.eval()
    input_list = ['v'+str(int(i//10)) for i in input_list]
    inputs = torch.LongTensor([tk.encode(input_list, pad_to_max_length=True, max_length=max_length)])
    return model(inputs)[0].resize(max_length, embedding_size)[0]
