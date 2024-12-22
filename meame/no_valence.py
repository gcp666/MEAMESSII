from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# ...

class Sample:
    def __init__(self, vid, speaker, label, text, audio, visual, sentence):
        self.vid = vid
        self.speaker = speaker
        self.label = label
        self.text = text
        self.audio = audio
        self.visual = visual
        self.sentence = sentence
        self.sbert_sentence_embeddings = self.encode_sentences(sentence)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_sentences(self, sentences):
        # 本地路径
        local_model_path = "E:\\hexin2\\COGMEN-main\\sentence-transformersparaphrase-distilroberta-base-v1"

        # 加载模型
        model = AutoModel.from_pretrained(local_model_path)

        # Tokenize sentences
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling. In this case, mean pooling.
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        return sentence_embeddings.numpy()


