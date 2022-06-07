import torch
from tqdm import tqdm
from dataset import collate_fn
from pylab import *
import matplotlib
import matplotlib.pyplot as plt
from random import shuffle
from math import floor
from sklearn import metrics
import os
# NOTE MODIFICATION (REFACTOR)
class MetricTracker(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, summed_val, n=1):
        self.val = summed_val / n
        self.sum += summed_val
        self.count += n
        self.avg = self.sum / self.count

    def print_metrics_binary(y_true, predictions, logging, verbose=1):
        predictions = np.array(predictions)
        if len(predictions.shape) == 1:
            predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))
        cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
        print(cf)
        if verbose:
            logging.info("confusion matrix:")
            logging.info(cf)
        cf = cf.astype(np.float32)

        acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
        prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
        prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
        rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
        rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
        auroc = metrics.roc_auc_score(y_true, predictions[:, 1])

        (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
        auprc = metrics.auc(recalls, precisions)
        minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

        if verbose:
            logging.info("accuracy = {0:.3f}".format(acc))
            logging.info("precision class 0 = {0:.3f}".format(prec0))
            logging.info("precision class 1 = {0:.3f}".format(prec1))
            logging.info("recall class 0 = {0:.3f}".format(rec0))
            logging.info("recall class 1 = {0:.3f}".format(rec1))
            logging.info("AUC of ROC = {0:.3f}".format(auroc))
            logging.info("AUC of PRC = {0:.3f}".format(auprc))
            logging.info("min(+P, Se) = {0:.3f}".format(minpse))

        return {"acc": acc,
                "prec0": prec0,
                "prec1": prec1,
                "rec0": rec0,
                "rec1": rec1,
                # most important to do model selection
                "auroc": auroc,
                "auprc": auprc,
                "minpse": minpse}

    # NOTE MODIFICATION (EMBEDDING)

def get_pretrained_weights(glove_path, corpus_vocab, embed_dim, device):
    """
    Returns 50002 words' pretrained weights in tensor
    :param glove_path: path of the glove txt file
    :param corpus_vocab: vocabulary from dataset
    :return: tensor (len(vocab), embed_dim)
    """
    save_dir = os.path.join(glove_path, 'glove_pretrained.pt')
    if os.path.exists(save_dir):
        return torch.load(save_dir, map_location=device)

    corpus_set = set(corpus_vocab)
    pretrained_vocab = set()
    glove_pretrained = torch.zeros(len(corpus_vocab), embed_dim)
    with open(os.path.join(glove_path, 'vectors.txt'), 'rb') as f:
        for l in tqdm(f):
            line = l.decode().split()
            if line[0] in corpus_set:
                pretrained_vocab.add(line[0])
                glove_pretrained[corpus_vocab.index(line[0])] = torch.from_numpy(np.array(line[1:]).astype(np.float))

        # handling 'out of vocabulary' words
        var = float(torch.var(glove_pretrained))
        for oov in corpus_set.difference(pretrained_vocab):
            glove_pretrained[corpus_vocab.index(oov)] = torch.empty(100).float().uniform_(-var, var)
        print("weight size:", glove_pretrained.size())
        torch.save(glove_pretrained, save_dir)
    return glove_pretrained


# NOTE MODIFICATION (FEATURE)
    # referenced to https://gist.github.com/ihsgnef/f13c35cd46624c8f458a4d23589ac768


def map_sentence_to_color(words, scores, sent_score):
    """
    :param words: array of words
    :param scores: array of attention scores each corresponding to a word
    :param sent_score: sentence attention score
    :return: html formatted string
    """

    sentencemap = matplotlib.cm.get_cmap('binary')
    wordmap = matplotlib.cm.get_cmap('OrRd')
    result = '<p><span style="margin:5px; padding:5px; background-color: {}">' \
        .format(matplotlib.colors.rgb2hex(sentencemap(sent_score)[:3]))
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    for word, score in zip(words, scores):
        color = matplotlib.colors.rgb2hex(wordmap(score)[:3])
        result += template.format(color, '&nbsp' + word + '&nbsp')
    result += '</span><p>'
    return result

    # NOTE MODIFICATION (FEATURE)


def bar_chart(categories, scores, graph_title='Prediction', output_name='prediction_bar_chart.png'):
    y_pos = arange(len(categories))

    plt.bar(y_pos, scores, align='center', alpha=0.5)
    plt.xticks(y_pos, categories)
    plt.ylabel('Attention Score')
    plt.title(graph_title)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.savefig(output_name)

    # NOTE MODIFICATION (FEATURE)


def visualize(model, dataset, doc):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    # Predicts, and visualizes one document with html file
    :param model: pretrained model
    :param dataset: news20 dataset
    :param doc: document to feed in
    :return: html formatted string for whole document
    """

    orig_doc = doc
    doc, num_sents, num_words = dataset.transform(doc)
    label = 0  # dummy label for transformation

    doc, label, doc_length, sent_length = collate_fn([(doc, label, num_sents, num_words)])
    score, word_att_weight, sentence_att_weight \
        = model(doc.to(device), doc_length.to(device), sent_length.to(device))

    # predicted = int(torch.max(score, dim=1)[1])
    classes = ['Cryptography', 'Electronics', 'Medical', 'Space']
    result = "<h2>Attention Visualization</h2>"

    bar_chart(classes, torch.softmax(score.detach(), dim=1).flatten().cpu(), 'Prediction')
    result += '<br><img src="prediction_bar_chart.png"><br>'
    for orig_sent, att_weight, sent_weight in zip(orig_doc, word_att_weight[0].tolist(),
                                                  sentence_att_weight[0].tolist()):
        result += map_sentence_to_color(orig_sent, att_weight, sent_weight)

    return result


def train_validation_test_split(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    file_list = list(filter(lambda file: file.startswith('patient'), all_files))
    shuffle(file_list)
    split = [0., 0.7, 0.2]
    training_index = floor(len(file_list) * split[0])
    validation_index = floor(len(file_list) * split[1])
    training = file_list[:training_index]
    validation = file_list[training_index:training_index + validation_index]
    testing = file_list[training_index + validation_index:]
    return training, validation, testing