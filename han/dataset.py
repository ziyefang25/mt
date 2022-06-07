

import re
import pandas as pd
from torch.utils.data import Dataset
import torch
from vocabulary import Vocabulary
# -*- encoding: utf-8 -*-
import re
import pandas as pd
from torch.utils.data import Dataset
import torch
from vocabulary import Vocabulary
import os
import codecs
import string
import random
import scispacy
import spacy
# Import the large dataset
import en_core_sci_lg
from scispacy.linking import EntityLinker

def to_string(list):
  str = " "
  for a in list:
    str = str + a + " "
  return str

class TextDataset(Dataset):
    """
       Loads a list of sentences into memory from a text file,
       split by newlines.
    """

    def __init__(self, input_files, file_name, label_file, vocab_file, data_dir, word_limit=50,
                 sentence_limit=200):
        self.data = []
        self.labels = []
        self.vocab_word_to_idx = Vocabulary(data_dir, vocab_file).word_to_idx
        self.word_limit = word_limit
        self.sentence_limit = sentence_limit
        self.names = input_files
        self.categories = []
        self.categories_total = []
        self.nlp = en_core_sci_lg.load()
        self.linker = EntityLinker(resolve_abbreviations=True, name="umls")
        self.nlp.add_pipe(self.linker)
        # label file is in the data dir
        label_data = pd.read_csv(os.path.join(data_dir, "result_csv_dead_los.csv"))
        label_data = label_data.fillna(0)
        mort = 0
        # mortality = []
        for input_file in input_files:
            if "(1)" in input_file:
                continue
            patient_name = input_file.split(".")[0]
            mortality_data = label_data.dead_after_disch_date[label_data.patient_id == patient_name]
            mortality = mortality_data.iloc[0]
            if mortality == -1:
                self.labels.append("1")
                # mortality.append(input_file)
                # print(input_file)
                mort = mort + 1
            else:
                self.labels.append("0")
        for input_file in input_files:
            if "(1)" in input_file:
                continue
            with codecs.open(os.path.join(data_dir, input_file), 'rU', encoding="utf-8") as f:
                lines = f.readlines()
                sentences = []
                sentence = []
                for i in range(len(lines)):
                    line = lines[i]
                    if i == 0:
                        next_line = line.strip()
                        category = next_line
                        self.categories.append(category)
                        continue
                    elif i == len(lines) - 1:
                        break
                    elif line == "\n":
                        category = lines[i + 1].strip()
                        self.categories.append(category)
                        lines.pop(i)
                        sentence_join = ' '.join(sentence)
                        sentences.append(sentence_join)
                        sentence = []
                    else:
                        line = line.strip()
                        token = re.split(" ", line)
                        token = ' '.join(token)
                        sentence.append(token)
                f.close()
            if sentences is None:
                continue
            self.data.append(sentences)
            self.categories_total.append(self.categories)
            self.categories = []
            self.med = 0

    def transform(self, text):
        # encode document
        #print("len", len(text))
        doc = [
            [self.vocab_word_to_idx[word] if word in self.vocab_word_to_idx.keys() else self.vocab_word_to_idx["<unk>"]
             for word in sent.split(" ")]
            for sent in text]
        doc = [sent[:self.word_limit] for sent in doc][:self.sentence_limit]
        num_sents = min(len(doc), self.sentence_limit)
        num_words = [min(len(sent), self.word_limit) for sent in doc]
        # skip erroneous ones
        if num_sents == 0:
            return None, -1, None

        return doc, num_sents, num_words

    def swap_noise(self, phrases):
        new_phrases = []

        for sent in phrases:
            if self.med > 144989:
                sent = ' '.join(sent)
                new_phrases.append(sent)
            else:
                new_phrase = []
                sent_2 = to_string(sent)
                doc = self.nlp(sent_2)
                # print(len(list_a))
                for entity in doc.ents:
                    outcome = random.random()
                    if outcome > 0.3:
                        break
                    else:
                        entity_text = entity.text
                        entity_text = entity_text.split(" ")
                        length = len(entity)
                        if self.med < 144989:
                        
                            if length == 1:
                                for word in sent:

                                    if word == entity_text[0]:
                                        if len(entity._.kb_ents) > 0:
                                            umls_ent = entity._.kb_ents[0]
                                            umls_data = self.linker.kb.cui_to_entity[umls_ent[0]]
                                            try:
                                                sent[sent.index(word)] = umls_data[2][0]
                                                self.med = self.med + 1

                                            except:
                                                a = 0
                            else:
                                orig_len = len(sent)
                                for n in range(0, orig_len):
                                    word = sent[n]
                                    if entity_text[0] == word:
                                        if len(entity._.kb_ents) > 0:
                                            umls_ent = entity._.kb_ents[0]
                                            umls_data = self.linker.kb.cui_to_entity[umls_ent[0]]
                                            try:
                                                ind = sent.index(word)
                                                #print("sent_before_long", sent)
                                                sent[sent.index(word)] = umls_data[2][0]
                                                self.med = self.med + length
                                                del sent[ind+1:ind+len(entity)]
                                                #print("sent_after_long", sent)
                                                break

                                            except:
                                                co = 0


                        else:
                            break
                sent = ' '.join(sent)
                new_phrases.append(sent)

        return new_phrases

    def __getitem__(self, i):
        label = self.labels[i]
        text = self.data[i]
        doc = [[word
                for word in sent.split(" ")]
               for sent in text]
        doc = [sent[:self.word_limit] for sent in doc][:self.sentence_limit]
        text = self.swap_noise(doc)
        name = self.names[i]
        categories = self.categories_total[i]
        # NOTE MODIFICATION (REFACTOR)
        doc, num_sents, num_words = self.transform(text)

        if num_sents == -1:
            return None

        num_medical = 0
        return doc, label, text, name, categories, num_sents, num_words, num_medical

    def __len__(self):
        return len(self.data)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def num_classes(self):
        return 1
        # return len(list(self.data.target_names))


def collate_fn(batch):
    batch = filter(lambda x: x is not None, batch)
    docs, labels, texts, name, categories, doc_lengths, sent_lengths, num_medical = list(zip(*batch))
    bsz = len(labels)
    batch_max_doc_length = max(doc_lengths)
    batch_max_sent_length = max([max(sl) if sl else 0 for sl in sent_lengths])
    docs_tensor = torch.zeros((bsz, batch_max_doc_length, batch_max_sent_length)).long()
    sent_lengths_tensor = torch.zeros((bsz, batch_max_doc_length)).long()
    for doc_idx, doc in enumerate(docs):
        doc_length = doc_lengths[doc_idx]
        sent_lengths_tensor[doc_idx, :doc_length] = torch.LongTensor(sent_lengths[doc_idx])
        for sent_idx, sent in enumerate(doc):
            sent_length = sent_lengths[doc_idx][sent_idx]
            docs_tensor[doc_idx, sent_idx, :sent_length] = torch.LongTensor(sent)
    labels_array = []
    for label in labels:
        labels_array.append(int(label))
    labels_tensor = torch.LongTensor(labels_array)
    return docs_tensor, labels_tensor, texts, name, categories, torch.LongTensor(doc_lengths), sent_lengths_tensor, num_medical

