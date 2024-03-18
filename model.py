import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForSequenceClassification, RobertaTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_backbone(name, output_attentions=False):
    if name == 'bert':
        from transformers import BertForMaskedLM, BertTokenizer
        backbone = BertForMaskedLM.from_pretrained('bert-base-uncased', output_attentions=output_attentions)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.name = 'bert-base-uncased'
    elif name == 'roberta':
        from transformers import RobertaForMaskedLM, RobertaTokenizer
        backbone = RobertaForMaskedLM.from_pretrained('roberta-base', output_attentions=output_attentions)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        tokenizer.name = 'roberta-base'
    elif name == 'roberta-large':
        from transformers import RobertaForMaskedLM, RobertaTokenizer
        backbone = RobertaForMaskedLM.from_pretrained('roberta-large', output_attentions=output_attentions)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        tokenizer.name = 'roberta-large'
    else:
        raise ValueError('No matching backbone network')

    return backbone, tokenizer

class Classifier(nn.Module):
    def __init__(self, args, backbone_name, backbone, n_classes, train_type='None'):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name
        self.dropout = nn.Dropout(0.1)
        self.n_classes = n_classes
        self.train_type = train_type

        if 'large' in backbone_name:
            self.n_dim = 1024
        else:
            self.n_dim = 768

        ##### Classifier for down-stream task
        self.net_cls = nn.Linear(self.n_dim, n_classes)

    def forward(self, x):
        out_cls_orig = self.backbone(x)[1]
        out_cls = self.dropout(out_cls_orig)
        out_cls = self.net_cls(out_cls)

        return out_cls

class Classifier_multi(nn.Module):
    def __init__(self, args, backbone_name, backbone, n_classes, train_type='None'):
        super(Classifier_multi, self).__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name
        self.dropout = nn.Dropout(0.1)
        self.n_classes = n_classes
        self.train_type = train_type

        if 'large' in backbone_name:
            self.n_dim = 1024
        else:
            self.n_dim = 768

        ##### Classifier for down-stream task
        self.net_cls1 = nn.Linear(self.n_dim, n_classes)
        self.net_cls2 = nn.Linear(self.n_dim, n_classes)
        self.net_cls3 = nn.Linear(self.n_dim, n_classes)
        self.net_cls4 = nn.Linear(self.n_dim, n_classes)
        self.net_cls5 = nn.Linear(self.n_dim, n_classes)

    def forward(self, x, inputs_embed=None, penuls=False):
        if self.backbone_name in ['bert', 'albert']:
            attention_mask = (x > 0).float()
        else:
            attention_mask = (x != 1).float()

        if inputs_embed is not None:
            out_cls_orig = self.backbone(None, attention_mask, inputs_embeds=inputs_embed)[1]
        else:
            out_cls_orig = self.backbone(x)[1]
        out_cls = self.dropout(out_cls_orig)
        out_cls1 = self.net_cls1(out_cls)
        out_cls2 = self.net_cls2(out_cls)
        out_cls3 = self.net_cls3(out_cls)
        out_cls4 = self.net_cls4(out_cls)
        out_cls5 = self.net_cls5(out_cls)

        return [out_cls1, out_cls2, out_cls3, out_cls4, out_cls5]

class Classifier_pref_ensemble(nn.Module):
    def __init__(self, args, backbone_name, backbone, n_classes, train_type='None'):
        super(Classifier_pref_ensemble, self).__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name
        self.dropout = nn.Dropout(0.1)
        self.n_classes = n_classes
        self.train_type = train_type

        if 'large' in backbone_name:
            self.n_dim = 1024
        else:
            self.n_dim = 768

        ##### Classifier for down-stream task
        self.net_cls = nn.Linear(self.n_dim, self.n_classes)

        ##### Classifier for measuring a preference between sentences
        self.net_prefs1, self.net_prefs2, self.net_prefs3 = [], [], []
        in_size = self.n_dim + self.n_dim
        self.dummy = torch.ones(1, self.n_dim).cuda()

        for j in range(2):
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
        if self.backbone_name in ['bert', 'albert']:
            attention_mask = (x > 0).float()
        else:
            attention_mask = (x != 1).float()

        if pref:
            _, out_cls_orig, out_cls_sent = self.backbone(x)
            out_cls = self.dropout(out_cls_orig)
            out_cls = self.net_cls(out_cls)

            out_cls_sent = out_cls_sent[:, 0, :]

            out_pref_1 = self.net_prefs1(torch.cat([out_cls_sent, y.unsqueeze(1) * self.dummy], dim=-1)).unsqueeze(0)
            out_pref_2 = self.net_prefs2(torch.cat([out_cls_sent, y.unsqueeze(1) * self.dummy], dim=-1)).unsqueeze(0)
            out_pref_3 = self.net_prefs3(torch.cat([out_cls_sent, y.unsqueeze(1) * self.dummy], dim=-1)).unsqueeze(0)
            out_pref = torch.cat([out_pref_1, out_pref_2, out_pref_3], dim=0)

            return out_cls, out_pref
        else:
            out_cls_orig = self.backbone(x)[1]
            out_cls = self.dropout(out_cls_orig)
            out_cls = self.net_cls(out_cls)

            return out_cls