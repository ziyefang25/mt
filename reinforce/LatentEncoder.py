import torch
import torch.nn as nn
from basicModel import WordLSTMEncoder, SentenceLSTMEncoder


class LatentEncoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 embed_dim: int = 200,
                 hidden_size_word: int = 200,
                 hidden_size_sent: int = 200,
                 output_size: int = 1,
                 dropout: float = 0.1,
                 layer: str = "rcnn",
                 nonlinearity: str = "sigmoid"
                 ):
        super(LatentEncoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.embed_layer = nn.Sequential(
            self.embeddings,
            nn.Dropout(p=dropout)
        )

        self.enc_layer_word = WordLSTMEncoder(embed_dim, hidden_size_word)
        self.enc_layer_sent = SentenceLSTMEncoder(hidden_size_word * 2, hidden_size_sent)
        enc_size = hidden_size_sent * 2

        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(enc_size, output_size),
            nn.Sigmoid() if nonlinearity == "sigmoid" else nn.LogSoftmax(dim=-1)
        )

    def init_embeddings(self, embeddings):
        """
        Initialized embedding layer with pretrained embeddings.
        :param embeddings: embeddings to init with
        """
        # NOTE MODIFICATION (EMBEDDING)
        self.embeddings.weight = nn.Parameter(embeddings)

    def freeze_embeddings(self, freeze=False):
        """
        Set whether to freeze pretrained embeddings.
        :param freeze: True to freeze weights
        """
        # NOTE MODIFICATION (EMBEDDING)
        self.embeddings.weight.requires_grad = not freeze

    def forward(self, docs, doc_lengths, sent_lengths, z):

        # Sort sents by decreasing order in sentence lengths
        # [B, S, W]
        docs = self.embed_layer(docs)
        # mask
        z = z.unsqueeze(-1)
        docs = docs * z

        x, _ = self.enc_layer_word(docs, doc_lengths, sent_lengths)
        #
        x,_ = torch.mean(x, dim=2)  # [B, S, H]
        print("size of sentence level")
        x, _ = self.enc_layer_sent(x, doc_lengths)
        x, _ = torch.mean(x, dim=1) # [B, H]
        x = self.output_layer(x)
        print(x)

        return x