import torch

from torch import nn
import torchvision.models as models
from transformers import BertConfig, BertModel
import os
import json


def load_config():
    cur_path = os.path.dirname(__file__)
    config_path = os.path.join(cur_path, "configs")
    with open(config_path) as f:
        return json.load(f)


def C1D(n: int = 1, dtype=torch.float16):
    _, l, ci, co, kernel, stride, padding = shape_configs["C1D"]
    inputs = torch.randn(n, ci, l, dtype=dtype)
    model = nn.Conv1d(ci, co, kernel, stride, padding)
    return model, inputs


def C2D(n: int = 1, dtype=torch.float16):
    _, h, w, ci, co, kernel, stride, padding = shape_configs["C2D"]
    inputs = torch.randn(n, ci, h, w, dtype=dtype)
    model = nn.Conv2d(ci, co, kernel, stride, padding)
    return model, inputs


def C3D(n: int = 1, dtype=torch.float16):
    _, d, h, w, ci, co, kernel, stride, padding = shape_configs["C3D"]
    inputs = torch.randn(n, ci, d, h, w, dtype=dtype)
    model = nn.Conv3d(ci, co, kernel, stride, padding)
    return model, inputs


def DEP(n: int = 1, dtype=torch.float16):
    _, h, w, c, kernel, stride, padding, factor = shape_configs["DEP"]
    inputs = torch.randn(n, c, h, w, dtype=dtype)
    model = nn.Conv2d(c, c * factor, kernel, stride, padding, groups=c)
    return model, inputs


def DIL(n: int = 1, dtype=torch.float16):
    _, h, w, ci, co, kernel, stride, padding, dilation = shape_configs["DIL"]
    inputs = torch.randn(n, ci, h, w, dtype=dtype)
    model = nn.Conv2d(ci, co, kernel, stride, padding, dilation=dilation)
    return model, inputs


def GMM_1024(b: int = 1, dtype=torch.float16):
    _, m, n, k = shape_configs["GMM-1024"]
    inputs = torch.randn(b, m, k, dtype=dtype)
    model = nn.Linear(k, n)
    return model, inputs


def GMM_4096(b: int = 1, dtype=torch.float16):
    _, m, n, k = shape_configs["GMM-4096"]
    inputs = torch.randn(b, m, k, dtype=dtype)
    model = nn.Linear(k, n)
    return model, inputs


def GRP(n: int = 1, dtype=torch.float16):
    _, h, w, ci, co, kernel, stride, padding, groups = shape_configs["GRP"]
    inputs = torch.randn(n, ci, h, w, dtype=dtype)
    model = nn.Conv2d(ci, co, kernel, stride, padding, groups=groups)
    return model, inputs


def T2D(n: int = 1, dtype=torch.float16):
    _, h, w, ci, co, kernel, stride, padding = shape_configs["T2D"]
    inputs = torch.randn(n, ci, h, w, dtype=dtype)
    model = nn.ConvTranspose2d(ci, co, kernel, stride, padding)
    return model, inputs


def CBR(n: int = 1, dtype=torch.float16):
    _, h, w, ci, co, kernel, stride, padding = shape_configs["CBR"]
    inputs = torch.randn(n, ci, h, w, dtype=dtype)
    model = nn.Sequential(
        nn.Conv2d(ci, co, kernel, stride, padding),
        nn.BatchNorm2d(co),
        nn.ReLU(),
    )
    return model, inputs


def TBG(b: int = 1, dtype=torch.float16):
    _, seq, head, dim = shape_configs["TBG"]
    query = torch.randn(b, seq, head, dim, dtype=dtype)
    value = torch.randn(b, seq, head, dim, dtype=dtype)

    class TGBModule(nn.Module):
        def forward(self, inputs):
            query, value = inputs
            # shape b, head, seq, dim
            query_T = torch.permute(query, (0, 2, 1, 3))
            # shape b, head, dim, seq
            value_T = torch.permute(value, (0, 2, 3, 1))
            return torch.matmul(query_T, value_T)

    return TGBModule(), (query, value)


def resnet(n: int = 1, dtype=torch.float16):
    shape = [n] + shape_configs["resnet"][1:]
    inputs = torch.rand(shape, dtype=dtype)
    model = models.resnet50()
    return model, inputs


def mobilenet_v2(n: int = 1, dtype=torch.float16):
    shape = [n] + shape_configs["mobilenet_v2"][1:]
    inputs = torch.rand(shape, dtype=dtype)
    model = models.mobilenet_v2()
    return model, inputs


def bert_large(n: int = 1, dtype=torch.float16):
    # dtype for bert large is not used
    configuration = BertConfig(
        num_hidden_layers=24,
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
        return_dict=False,
    )

    shape = [n] + shape_configs["bert_large"][1:]
    model = BertModel(configuration)
    inputs = torch.randint(10000, shape, dtype=torch.int32)
    return model, inputs


def vit(n: int = 1, dtype=torch.float16):
    shape = [n] + shape_configs["vit"][1:]
    inputs = torch.rand(shape, dtype=dtype)
    model = models.vit_l_32()
    return model, inputs


shape_configs = load_config()


WORKLOADS = {
    "C1D": C1D,
    "C2D": C2D,
    "C3D": C3D,
    "DEP": DEP,
    "DIL": DIL,
    "GMM-1024": GMM_1024,
    "GMM-4096": GMM_4096,
    "GRP": GRP,
    "T2D": T2D,
    "CBR": CBR,
    "TBG": TBG,
    "resnet_50": resnet,
    "mobilenet_v2": mobilenet_v2,
    "bert_large": bert_large,
    "vit": vit,
}
