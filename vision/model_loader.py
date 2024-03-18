from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier_pref_ensemble(nn.Module):
    def __init__(self, args):
        super(Classifier_pref_ensemble, self).__init__()

        if args.finetune:
            self.backbone = models.resnet18(pretrained=True)
        else:
            self.backbone = models.resnet18(pretrained=False)
        self.n_classes = 2
        self.n_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        ##### Classifier for down-stream task
        self.net_cls = nn.Linear(self.n_dim, self.n_classes)

        ##### Classifier for measuring a similarity between sentences
        self.net_prefs1, self.net_prefs2, self.net_prefs3 = [], [], []
        in_size = self.n_dim + self.n_dim
        self.dummy = torch.ones(1, self.n_dim).cuda()

        for j in range(1):
            self.net_prefs1.append(nn.Linear(in_size, self.n_dim))
            self.net_prefs1.append(nn.ReLU())
            self.net_prefs2.append(nn.Linear(in_size, self.n_dim))
            self.net_prefs2.append(nn.ReLU())
            self.net_prefs3.append(nn.Linear(in_size, self.n_dim))
            self.net_prefs3.append(nn.ReLU())
            in_size = self.n_dim
        self.net_prefs1.append(nn.Linear(in_size, 1))
        self.net_prefs2.append(nn.Linear(in_size, 1))
        self.net_prefs3.append(nn.Linear(in_size, 1))

        self.net_prefs1.append(nn.Tanh())
        self.net_prefs2.append(nn.Tanh())
        self.net_prefs3.append(nn.Tanh())
    
        self.net_prefs1 = nn.Sequential(*self.net_prefs1).cuda()
        self.net_prefs2 = nn.Sequential(*self.net_prefs2).cuda()
        self.net_prefs3 = nn.Sequential(*self.net_prefs3).cuda()

    def forward(self, x, y=None, pref=False):
        if pref:
            penuls = self.backbone(x)
            out_cls = self.net_cls(penuls)

            out_cls_sent = penuls
            out_pref_1 = self.net_prefs1(torch.cat([out_cls_sent, y.unsqueeze(1) * self.dummy], dim=-1)).unsqueeze(0)
            out_pref_2 = self.net_prefs2(torch.cat([out_cls_sent, y.unsqueeze(1) * self.dummy], dim=-1)).unsqueeze(0)
            out_pref_3 = self.net_prefs3(torch.cat([out_cls_sent, y.unsqueeze(1) * self.dummy], dim=-1)).unsqueeze(0)
            out_pref = torch.cat([out_pref_1, out_pref_2, out_pref_3], dim=0)

            return out_cls, out_pref
        else:
            out_cls = self.net_cls(self.backbone(x))
            return out_cls