from types import SimpleNamespace

config = SimpleNamespace()
config.embedding_size = 512 # embedding size of model
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128 # batch size per GPU
config.lr = 0.1
config.root_dir = "/dataset"
config.output = "/output"
config.global_step = 0 # step to resume
config.s = 64.0
config.m = 0.35
config.std = 0.05
config.SE = False
config.use_augmentation = True
config.train_samples_file = "trainsamples_normal_10.txt"

config.loss="CosFace"

# type of network to train [iresnet100 | iresnet50]
config.network = "iresnet50"

config.rec = "/benchmarks"
config.num_classes = 10572
config.num_epoch = 40
config.warmup_epoch = -1
config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
config.eval_step = 2000


def lr_step_func(epoch):
    return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
        [m for m in [24, 30, 36] if m - 1 <= epoch])
    # [20, 26, 30] [22, 30, 40]
config.lr_func = lr_step_func
