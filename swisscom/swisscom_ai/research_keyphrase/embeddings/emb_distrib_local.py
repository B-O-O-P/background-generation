# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
# Authors: Kamil Bennani-Smires, Yann Savary

import numpy as np
import sys

sys.path.append("..")
from sentence_transformers import SentenceTransformer
from swisscom.swisscom_ai.research_keyphrase.embeddings.emb_distrib_interface import EmbeddingDistributor
import torch
import transformers as ppb

class EmbeddingDistributorLocal(EmbeddingDistributor):
    """
    Concrete class of @EmbeddingDistributor using a local installation of sent2vec
    https://github.com/epfml/sent2vec
    
    """

    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        # self.model = sent2vec.Sent2vecModel()
        # self.model.load_model(fasttext_model)

    def get_tokenized_sents_embeddings(self, sents):
        """
        @see EmbeddingDistributor
        """
        for sent in sents:
            if '\n' in sent:
                raise RuntimeError('New line is not allowed inside a sentence')

        return self.model.encode(sentences=sents)
