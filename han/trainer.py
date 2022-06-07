import torch
from test import Tester
from validation import Validate
from utils import MetricTracker
import logging
import os
import pickle

class Trainer:
    def __init__(self, config, model, optimizer, criterion, dataloader):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = next(self.model.parameters()).device
        self.losses = MetricTracker()
        self.accs = MetricTracker()
        self.tester = Tester(self.config, self.model)
        self.validate = Validate(self.config, self.model)
        self.results = []
        self.best_val_auc = 0
        self.steps = config.steps

    def train(self):
        if not os.path.exists(os.path.dirname('best_model')):
            os.makedirs('best_model', exist_ok=True)

        step = 0
        for epoch in range(self.config.num_epochs):
            self.model.train()
            self.losses.reset()
            for batch_idx, (docs, labels, doc_lengths, sent_lengths) in enumerate(self.dataloader):
                batch_size = labels.size(0)
                docs = docs.to(self.device)  # (batch_size, padded_doc_length, padded_sent_length)
                labels = labels.to(self.device)  # (batch_size)
                sent_lengths = sent_lengths.to(self.device)  # (batch_size, padded_doc_length)
                doc_lengths = doc_lengths.to(self.device)  # (batch_size)

                # (n_docs, n_classes), (n_docs, max_doc_len_in_batch, max_sent_len_in_batch), (n_docs, max_doc_len_in_batch)
                scores, word_att_weights, sentence_att_weights = self.model(docs, doc_lengths, sent_lengths)

                # NOTE MODIFICATION (BUG)
                self.optimizer.zero_grad()

                loss = self.criterion(scores, labels)
                loss.backward()

                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                # NOTE MODIFICATION (BUG): clip grad norm should come before optimizer.step()
                self.optimizer.step()

                # Compute accuracy
                predictions = scores.max(dim=1)[1]

                self.losses.update(loss.item(), batch_size)

                # Every 100 steps we evaluate the model and report progress.
                if step % self.steps == 0:
                    logging.info("epoch (%d) step %d: training loss = %.2f" %
                                 (epoch, step, self.losses.avg))

                step += 1

            # NOTE MODIFICATION (TEST)
            metrics_results = self.validate.eval()
            self.results.append(metrics_results)
            if metrics_results['auroc'] > self.best_val_auc:
                best_val_auc = metrics_results['auroc']
                # save best model in disk
                torch.save({
                    'epoch': epoch,
                    'model': self.model,
                    'optimizer': self.optimizer,
                }, 'best_model/model.pth.tar')
                logging.info('best model AUC of ROC = %.3f' % (best_val_auc))
                logging.info("Finished epoch %d" % (epoch))

            pickle.dump(self.results, open(os.path.join(self.config.output_dir, 'metrics.pkl'), "wb"))


