from dataset import TextDataset, collate_fn
from utils import *
import os, sys
import webbrowser


class Tester:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.device = next(self.model.parameters()).device

        self.dataset = TextDataset(config.test_files, config.label_file, config.vocab_file)
        print(self.dataset.data)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=config.batch_size, shuffle=False,
                                                      collate_fn=collate_fn)

        self.accs = MetricTracker()

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            self.accs.reset()

            for (docs, labels, doc_lengths, sent_lengths) in self.dataloader:
                batch_size = labels.size(0)

                docs = docs.to(self.device)
                labels = labels.to(self.device)
                doc_lengths = doc_lengths.to(self.device)
                sent_lengths = sent_lengths.to(self.device)

                scores, word_att_weights, sentence_att_weights = self.model(docs, doc_lengths, sent_lengths)

                predictions = scores.max(dim=1)[1]
                correct_predictions = torch.eq(predictions, labels).sum().item()
                acc = correct_predictions

                self.accs.update(acc, batch_size)
            self.best_acc = max(self.best_acc, self.accs.avg)

            print('Test Average Accuracy: {acc.avg:.4f}'.format(acc=self.accs))


if __name__ == "__main__":
    if not os.path.exists("best_model/model.pth.tar"):
        print("Visualization requires pretrained model to be saved under ./best_model.\n")
        print("Please run 'python train.py <args>'")
        sys.exit()

    checkpoint = torch.load("best_model/model.pth.tar")
    model = checkpoint['model']
    model.eval()
    test_files = train_validation_test_split("testcomebine")[2]
    dataset = TextDataset(test_files, "vectors.txt", "result_csv_dead_loss.csv", word_limit=50, sentence_limit=200)
    #4 or 5 patients from the validation
    doc = "13 patient / test information : indication : coronary artery disease"

    result = visualize(model, dataset, doc)

    with open('result.html', 'w') as f:
        f.write(result)

    webbrowser.open_new('file://'+os.getcwd()+'/result.html')