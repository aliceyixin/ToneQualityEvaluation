import pytorch_lightning as pl
import torch
import os
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.transforms as transforms
from TQNet_instruments import TQNet
from train_sax import saxDataset
from torch.utils.data import DataLoader
import numpy as np
import librosa


os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

def to_mfcc(x):
    x = x.T
    mfcc = librosa.feature.mfcc(sr=48000, S=x, n_mfcc=13, dct_type=2, norm='ortho')
    mfcc_delta = librosa.feature.delta(mfcc, width=9, order=1, axis=-1)
    mfcc_delta2 = librosa.feature.delta(mfcc, width=9, order=2, axis=-1)
    mfccs = np.vstack((mfcc, mfcc_delta, mfcc_delta2))
    return mfccs


class clarinetDataset(torch.utils.data.Dataset):
    """ clarine_gs Dataset extraction """
    def __init__(self, csv_file, feature_path, feature_type, augment, transforms):
        self.feature_type = feature_type
        self.transforms = transforms
        labels = []
        filenames = []
        if augment is False:
            with open(csv_file, 'r') as f:
                next(f)  # skip first line
                content = f.readlines()
                for x in content:
                    row = x.strip('\n').split(',')
                    name = '_'.join(row[1].split('/')[-3:])
                    filepath = os.path.join(args.feature_path + '/' + name[:-4] + ".npy")
                    filenames.append(filepath)
                    label = int(row[2])
                    if label == 2:
                        label = 1
                    labels.append(label)
        else:
            with open(csv_file, 'r') as f:
                next(f)  # skip first line
                content = f.readlines()
                for x in content:
                    row = x.strip('\n').split(',')
                    label = int(row[2])
                    if label == 2:
                        label = 1
                    filenames.append(os.path.join(feature_path, row[1].replace(".wav", ".npy")))
                    labels.append(label)
                    filenames.append(os.path.join(feature_path, row[1].replace(".wav", "_pu.npy")))
                    labels.append(label)
                    filenames.append(os.path.join(feature_path, row[1].replace(".wav", "_pd.npy")))
                    labels.append(label)
                    filenames.append(os.path.join(feature_path, row[1].replace(".wav", "_lu.npy")))
                    labels.append(label)
                    filenames.append(os.path.join(feature_path, row[1].replace(".wav", "_ld.npy")))
                    labels.append(label)
        self.datalist = filenames
        self.feature_path = feature_path
        self.labels = labels
        # print("Total data: {}".format(len(self.datalist)))
        # print("Total label: {}".format(len(self.labels)))

    def __len__(self):
        """ set the len(object) funciton """
        return len(self.datalist)

    def __getitem__(self, idx):
        """
        Function to extract the spectrogram samples and labels from the audio dataset.
        """
        data_path = self.datalist[idx]
        x = np.load(data_path)
        if self.feature_type == 'mfcc':
            x = to_mfcc(x)
        else:
            pass
        label = np.asarray(int(self.labels[idx]))

        x = np.reshape(x, [1, x.shape[0], x.shape[1]])
        x = torch.from_numpy(x).type(torch.FloatTensor)
        label = torch.from_numpy(label).type(torch.LongTensor)

        return x, label


class DecayLearningRate(pl.Callback):
    def __init__(self):
        self.old_lrs = []

    def on_train_start(self, trainer, pl_module):
        # track the initial learning rates
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            group = []
            for param_group in optimizer.param_groups:
                group.append(param_group["lr"])
            self.old_lrs.append(group)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            old_lr_group = self.old_lrs[opt_idx]
            new_lr_group = []
            for p_idx, param_group in enumerate(optimizer.param_groups):
                old_lr = old_lr_group[p_idx]
                new_lr = old_lr * 0.99
                new_lr_group.append(new_lr)
                param_group["lr"] = new_lr
            self.old_lrs[opt_idx] = new_lr_group


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    instrument = 'clarinet'  # cello trumpet cello  flute
    parser.add_argument("--train_dir", default="../goodsoundData/gs_{}_train.csv".format(instrument))
    parser.add_argument("--eval_dir", default="../goodsoundData/gs_{}_val.csv".format(instrument))
    parser.add_argument("--test_dir", default="../goodsoundData/gs_{}_test.csv".format(instrument))
    parser.add_argument("--use_cuda", default=False)
    parser.add_argument("--gpus", default=1)
    parser.add_argument("--batch_size", default=20)
    parser.add_argument("--learning_rate", default=1e-5)
    parser.add_argument("--epochs", default=5)
    parser.add_argument("--train_augment", default=False)
    parser.add_argument("--test_augment", default=False)
    parser.add_argument("--feature_path", default="../goodsoundData/{}/features/melspec".format(instrument))  # LPRmelspec
    parser.add_argument("--feature_type", default='spec')
    args = parser.parse_args()

    feature_path = Path(args.feature_path)

    device = torch.device("cuda" if args.use_cuda else "cpu")

    data_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    clarinet_train = clarinetDataset(
        csv_file=args.train_dir,
        feature_path=feature_path,
        feature_type=args.feature_type,
        augment=args.train_augment,
        transforms=data_transform,
    )

    clarinet_val = clarinetDataset(
        csv_file=args.eval_dir,
        feature_path=feature_path,
        feature_type=args.feature_type,
        augment=args.train_augment,
        transforms=data_transform,
    )
    clarinet_test = clarinetDataset(
        csv_file=args.test_dir,
        feature_path=feature_path,
        feature_type=args.feature_type,
        augment=args.test_augment,
        transforms=data_transform,
    )
    val_loader = DataLoader(clarinet_val, batch_size=args.batch_size, shuffle=False, num_workers=1)  # 
    test_loader = DataLoader(clarinet_test, batch_size=int(args.batch_size), shuffle=False, num_workers=1)  # 

    # see data form
    print("Total train data: {}".format(len(clarinet_train)))
    print("Total val train data: {}".format(len(clarinet_val)))

    model = TQNet(
        clarinet_train, clarinet_val, clarinet_test, args.batch_size, args.learning_rate
        ).load_from_checkpoint('PretrainedModels/TQNetsax-melspec-epoch=29-augT-0225.ckpt')


    for param in model.encoder.parameters():
        param.requires_grad = False

    logger = TensorBoardLogger(
        save_dir=".",
        name="sax2instr_gs_logs",
        log_graph=True,
    )
    metrics = {'feature_path': args.feature_path, 'feature_type': args.feature_type,
               'train_augment': args.train_augment, 'epochs': args.epochs,
               'batch_size': args.batch_size, 'learning_rate': args.learning_rate}
    logger.log_hyperparams(metrics)

    examples = iter(val_loader)
    example_data, example_targets = examples.next()
    logger.log_graph(model, input_array=example_data)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        gpus=args.gpus,
        distributed_backend="ddp",
        progress_bar_refresh_rate=1,
        callbacks=[
            DecayLearningRate(),
        ],
    )

    trainer.fit(model)
    trainer.test(model)