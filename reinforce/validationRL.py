from utils import *
import sys
from utils import MetricTracker
import webbrowser
import os.path
import logging
from dataset import TextDataset


class Validate:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.device = next(self.model.parameters()).device
        self.dataset = TextDataset(config.validation_files, config.label_file, config.vocab_file, config.data_dir)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=config.batch_size, shuffle=False,
                                                      collate_fn=collate_fn)

    def eval(self):
        y_true = []
        predictions_all = []
        self.model.eval()
        with torch.no_grad():
            for (docs, labels, doc_lengths, sent_lengths) in self.dataloader:
                docs = docs.to(self.device)
                labels = labels.to(self.device)
                doc_lengths = doc_lengths.to(self.device)
                sent_lengths = sent_lengths.to(self.device)
                scores = self.model(docs, doc_lengths, sent_lengths)
                predictions = scores
                predictions_all += [p.item() for p in predictions]  # y_hat_class.squeeze()
                y_true += [y.item() for y in labels]
            metrics_results = MetricTracker.print_metrics_binary(y_true, predictions_all, logging)
            return metrics_results


if __name__ == "__main__":
    if not os.path.exists("best_model/model.pth.tar"):
        print("Visualization requires pretrained model to be saved under ./best_model.\n")
        print("Please run 'python train.py <args>'")
        sys.exit()

    checkpoint = torch.load("best_model/model.pth.tar")
    model = checkpoint['model']
    model.eval()
    data_dir = "D:/srp/30patients"
    train_files, validation_files, test_files = train_validation_test_split(data_dir)
    label_file = os.path.join(data_dir,"result_csv_dead_los.csv")
    vocab_file= os.path.join(data_dir,"vectors.txt")
    dataset = TextDataset(validation_files, label_file, vocab_file, data_dir)
    # 4 or 5 patients from the validation
    doc = ["r fem sheath in place in groin", "waveform sharp", "no bleeding or hematoma at site", "dsd intact",
           "l aline patent", "site wnl , waveform sharp", "ls clear bilat", "o2 weaned to 2l nc", "sats 90 - 100 %",
           "abd soft with + bs", "no stool", "foley patent , draining qs clear yellow urine",
           "plan : monitor neuro status for changed", "maintain sbp < 100", "monitor sheath site , r groin",
           "? d / c to floor tomorrow"]
    result = visualize(model, dataset, doc)

    with open('result.html', 'w') as f:
        f.write(result)

    webbrowser.open_new('file://' + os.getcwd() + '/result.html')
