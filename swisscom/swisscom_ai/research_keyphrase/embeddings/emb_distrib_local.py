# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
# Authors: Kamil Bennani-Smires, Yann Savary

import numpy as np
import sys

sys.path.append("..")
from swisscom.swisscom_ai.research_keyphrase.embeddings.emb_distrib_interface import EmbeddingDistributor
import torch
import transformers as ppb

class EmbeddingDistributorLocal(EmbeddingDistributor):
    """
    Concrete class of @EmbeddingDistributor using a local installation of sent2vec
    https://github.com/epfml/sent2vec
    
    """

    def __init__(self, pretrained_weights='bert-base-uncased'):
        configuration = ppb.BertConfig()
        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel(configuration),
                                                            ppb.BertTokenizer,
                                                            pretrained_weights)
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights)
        # self.model = sent2vec.Sent2vecModel()
        # self.model.load_model(fasttext_model)

    def get_tokenized_sents_embeddings(self, sents):
        """
        @see EmbeddingDistributor
        """
        for sent in sents:
            if '\n' in sent:
                raise RuntimeError('New line is not allowed inside a sentence')

        tokenized = list(map(lambda x: self.tokenizer.encode(x, add_special_tokens=True), sents))

        max_len = 0
        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
        input_ids = torch.tensor(np.array(padded)).type(torch.LongTensor)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)

        vectors = last_hidden_states[0][:, 0, :].numpy()

        return vectors
