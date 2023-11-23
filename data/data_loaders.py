import os
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

DATASET_PATHS = {
    "mit-states": os.path.join(DIR_PATH, "../data/mit-states"),
    "ut-zappos": os.path.join(DIR_PATH, "../data/ut-zappos"),
    "cgqa": os.path.join(DIR_PATH, "../cgqa"),
    "coco": os.path.join(DIR_PATH, "../data/mscoco/train2017")
}

def get_dataloader(config, dataset, is_train = True):
    
    if is_train:
        sampler = RandomSampler(dataset)
        batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)
    else:
        sampler = SequentialSampler(dataset)
        batch_size = config.per_gpu_eval_batch_size * max(1, config.n_gpu)

    dataloader = DataLoader(dataset, sampler=sampler, 
            batch_size=batch_size, num_workers=config.num_workers)

    return dataloader

