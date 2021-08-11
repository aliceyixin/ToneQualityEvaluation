
import pytorch_lightning as pl
import torch
# from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.metrics.functional import accuracy
# from sklearn.metrics import accuracy_score, classification_report
feature_dim = 80  # 39 or 80
time_dim = 300  # cla 80, sax 150


class TimbreEncoder(pl.LightningModule):
    def __init__(self, drop_connect_rate=0.1):
        super(TimbreEncoder, self).__init__()
        self.save_hyperparameters()

        self.convlayer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(time_dim, 1), dilation=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=(1, 2)),  # , stride=3

            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),

            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2)),  ## VQual（1,2）
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            # torch.nn.Dropout2d(p=0.2),
        )
        self.convlayer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(time_dim, 5), dilation=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=(1, 2)),

            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2)),  ## VQual（1,2）
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            # torch.nn.MaxPool2d(kernel_size=2),  # , stride=3
            # torch.nn.Dropout2d(p=0.2)
        )
        self.convlayer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, feature_dim), dilation=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=(2, 1)),

            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 1)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            # torch.nn.MaxPool1d(kernel_size=2),  # , stride=3
            # torch.nn.Dropout2d(p=0.2)
        )
        self.convlayer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, feature_dim), dilation=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=(2, 1)),

            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 1)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            # torch.nn.MaxPool1d(kernel_size=2),  # , stride=3
            # torch.nn.Dropout2d(p=0.2)
        )

        self.fc = torch.nn.Sequential(
            # cla: spec=4416, mfcc=7360
            # cla2sax spec=518336
            # sax: spec=11456 mfcc=11840
            # torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # torch.nn.Dropout2d(p=0.5),
            torch.nn.Linear(in_features=11456, out_features=512),
            torch.nn.ReLU(True),
            # torch.nn.Dropout2d(p=0.2),
            # torch.nn.Linear(in_features=2048, out_features=512),
        )

    def forward(self, x):
        # print(x.shape) 20,80,80
        # x = x.unsqueeze(1)
        c1 = self.convlayer1(x)
        c2 = self.convlayer2(x)
        c3 = self.convlayer3(x)
        c4 = self.convlayer4(x)
        c1 = c1.view(c1.size(0), -1)
        c2 = c2.view(c2.size(0), -1)
        c3 = c3.view(c3.size(0), -1)
        c4 = c4.view(c4.size(0), -1)
#         print("output size:", c1.shape,c2.shape,c3.shape,c4.shape)
        x = torch.cat([c1, c2, c3, c4], dim=1)
        x = x.view(x.size(0), -1)
        # print("layer nodes before fc", x.shape)
        x = self.fc(x)
        # y = x.squeeze(3).squeeze(2)
        return x


class TQNet(pl.LightningModule):

    def __init__(self, data_train, data_val, data_test, batch_size, learning_rate):
        # , feature_path, feature_type, train_dir, eval_dir, train_augment, test_augment
        super().__init__()
        self.save_hyperparameters()

        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test

        self.lr = learning_rate
        self.batch_size = batch_size
        self.p = 0.1
        self.do = torch.nn.Dropout(p=self.p)

        self.encoder = TimbreEncoder(drop_connect_rate=self.p)

        self.g = torch.nn.Linear(512, 256)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=256)
        self.linear = torch.nn.Linear(256, 2, bias=False)

    def forward(self, x):
        x = self.do(self.encoder(x))
        x = self.do(self.g(x))
        x = self.do(torch.tanh(self.layer_norm(x)))
        x = self.linear(x)
        x = torch.log_softmax(x, dim=1)
        return x  # F.log_softmax(t, dim=1)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('train_loss', loss)
        self.log("train_acc", acc)
        # f1 = f1_loss(y, predicted)
        # self.log("f1_loss_train", f1)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss)
        self.log("valid_acc", acc)
        # f1 = f1_loss(y, preds)
        # self.log("f1_loss_val", f1)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('test_loss', loss)
        self.log("test_acc", acc)
        # f1 = f1_loss(y, preds)
        # self.log("f1_loss_test", f1)
        return loss

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=int(self.batch_size), shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=int(self.batch_size), shuffle=False, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=int(self.batch_size), shuffle=False, num_workers=4)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        return optimizer

