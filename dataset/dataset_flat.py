from dataset.dataset_sequential import DatasetSequential, DatasetSequentialAugmented


class DatasetFlat(DatasetSequential):
    def __getitem__(self, idx):
        x_seq, y = super().__getitem__(idx)

        return x_seq.flatten(), y

class DatasetFlatAugmented(DatasetSequentialAugmented):
    def __getitem__(self, idx):
        x_seq, y = super().__getitem__(idx)

        return x_seq.flatten(), y
