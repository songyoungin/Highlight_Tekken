import torch
from torch import nn
from torch.autograd import Variable
from dataloader import get_loader

# pretrained CNN network
class C2D(nn.Module):
    def __init__(self):
        super(C2D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            nn.Conv2d(64, 128, 4, 2, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            nn.Conv2d(256, 256, 4, 2, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Conv2d(256, 1, 1, 1, 0)
        self.dropout = nn.Dropout(p=0.5)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # 115, 256, 1, 1
        out = self.fc(out)
        out = self.dropout(out)
        out = self.sig(out)
        return out

class GRU(nn.Module):
    def __init__(self, c2d):
        super(GRU, self).__init__()
        self.c2d = c2d
        self.c2d.cuda()
        self.sig = nn.Sigmoid().cuda()

    def forward(self, x):
        c2d_out = self.c2d(x) # frame, 1, 1, 1
        c2d_out = c2d_out.squeeze(1) # frame, 1, 1
        # print("C2D output shape:", c2d_out.shape)
        # print("C2D output:", c2d_out)

        self.gru = nn.GRU(input_size=1, hidden_size=1).cuda()
        h0 = torch.randn(1, 1, 1).cuda()

        gru_out, _ = self.gru(c2d_out, h0)
        gru_out = gru_out.squeeze()

        out = self.sig(gru_out)

        # print("GRU output shape:", gru_out.shape)
        # print("GRU output:", gru_out)

        return out

# if __name__ == "__main__":
#     root = "C:\\Users\\USER\Desktop\PROGRAPHY DATA_ver3"
#
#     hLoader, rLoader, tLoader = get_loader(root + '\HV',
#                                             root + '\RV',
#                                             root + '\\testRV', 1)
#
#     c2d = C2D().cuda()
#     criteria = nn.BCELoss()
    # opt = torch.optim.SGD(filter(lambda p: p.requires_grad, c2d.parameters()),
    #                       lr=0.05,
    #                       weight_decay=0.005)

    # # train on cnn
    # for idx, video in zip(enumerate(hLoader)):
    #     video = video[0].cuda()
    #     video = Variable(video).cuda()
    #
    #     out = c2d(video)
    #     target = torch.on
    #     loss = criteria(out, )


