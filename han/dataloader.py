from torch.utils.data import DataLoader
from dataset import collate_fn
from samplerFourDim import ImbalancedDatasetSampler


class MyDataLoader(DataLoader):

    def __init__(self, dataset, batch_size):

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'pin_memory': True,
            'collate_fn': collate_fn
        }
        super().__init__(**self.init_kwargs)
