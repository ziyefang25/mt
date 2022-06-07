import os, sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import HierarchicalAttentionNetwork
from dataset import TextDataset
from dataloader import MyDataLoader
from trainer import Trainer
from utils import get_pretrained_weights, train_validation_test_split
from vocabulary import Vocabulary


def train(config, device):
    vocab_word_to_idx = Vocabulary(config.vocab_file).word_to_idx
    vocab = list(vocab_word_to_idx.keys())
    dataset = TextDataset(config.train_files, config.label_file, config.vocab_file)
    dataloader = MyDataLoader(dataset, config.batch_size)
    model = HierarchicalAttentionNetwork(
        num_classes=dataset.num_classes,
        vocab_size=len(vocab[0]),
        embed_dim=config.embed_dim,
        word_gru_hidden_dim=config.word_gru_hidden_dim,
        sent_gru_hidden_dim=config.sent_gru_hidden_dim,
        word_gru_num_layers=config.word_gru_num_layers,
        sent_gru_num_layers=config.sent_gru_num_layers,
        word_att_dim=config.word_att_dim,
        sent_att_dim=config.sent_att_dim,
        use_layer_norm=config.use_layer_norm,
        dropout=config.dropout).to(device)

    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)

    # NOTE MODIFICATION (BUG)
    # criterion = nn.NLLLoss(reduction='sum').to(device) # option 1
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)  # option 2

    # NOTE MODIFICATION (EMBEDDING)
    if config.pretrain:
        weights = get_pretrained_weights("C:/Users/ziyefang96/PycharmProjects/machinelearning/bin/HAN", vocab, config.embed_dim, device)
        model.sent_attention.word_attention.init_embeddings(weights)
    model.sent_attention.word_attention.freeze_embeddings(config.freeze)

    trainer = Trainer(config, model, optimizer, criterion, dataloader)
    trainer.train()


if __name__ == '__main__':
    train_files, validation_files, test_files = train_validation_test_split("C:/Users/ziyefang96/PycharmProjects/machinelearning/bin/HAN/testcombine")
    parser = argparse.ArgumentParser(description='Bug squash for Hierarchical Attention Networks')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=5)

    parser.add_argument("--embed_dim", type=int, default=200)
    parser.add_argument("--word_gru_hidden_dim", type=int, default=100)
    parser.add_argument("--sent_gru_hidden_dim", type=int, default=100)
    parser.add_argument("--word_gru_num_layers", type=int, default=1)
    parser.add_argument("--sent_gru_num_layers", type=int, default=1)
    parser.add_argument("--word_att_dim", type=int, default=200)
    parser.add_argument("--sent_att_dim", type=int, default=200)
    parser.add_argument("--label_file", type=str, default="result_csv_dead_los.csv")
    parser.add_argument("--vocab_file", type=str, default="vectors.txt")
    parser.add_argument("--train_files", type=list, default=train_files)
    parser.add_argument("--validation_files", type=list, default=validation_files)
    parser.add_argument("--test_files", type=list, default=test_files)
    parser.add_argument('--steps', type=int, default=100,help='perfom evaluation and model selection on validation default:100')
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='C:/Users/ziyefang96/PycharmProjects/machinelearning/bin/HAN')
    # NOTE MODIFICATION (EMBEDDING)
    parser.add_argument("--pretrain", type=bool, default=True)
    parser.add_argument("--freeze", type=bool, default=False)

    # NOTE MODIFICATION (FEATURES)
    parser.add_argument("--use_layer_norm", type=bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.1)

    config = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NOTE MODIFICATION (FEATURE)
    if not os.path.exists(os.path.dirname('best_model')):
        os.makedirs('best_model', exist_ok=True)

    train(config, device)