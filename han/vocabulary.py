UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

import os


class Vocabulary:
    """
        Creates a vocabulary from a word2vec file.
    """

    def __init__(self, vocab_file_path, vocab_file):
        self.word_to_idx, self.idx_to_word = self.from_data(vocab_file_path, vocab_file)

    def __getitem__(self, key):
        return self.word_to_idx[key] if key in self.word_to_idx else self.word_to_idx[UNK_TOKEN]

    def word(self, idx):
        return self.idx_to_word[idx]

    def size(self):
        return len(self.word_to_idx)

    def from_data(self, vocab_file_path, input_file):
        word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1, SOS_TOKEN: 2, EOS_TOKEN: 3}
        idx_to_word = {0: PAD_TOKEN, 1: UNK_TOKEN, 2: SOS_TOKEN, 3: EOS_TOKEN}
        with open(os.path.join(vocab_file_path, input_file), 'rb') as f:
            for l in f:
                line = l.decode().split()
                token = line[0]
                if token not in word_to_idx:
                    idx = len(word_to_idx)
                    word_to_idx[token] = idx
                    idx_to_word[idx] = token
        return word_to_idx, idx_to_word
