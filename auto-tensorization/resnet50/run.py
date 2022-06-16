import pickle
import sys
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import tvm
from tvm import relay
import tvm.relay.testing
from tvm.relay.op.contrib.cutlass import partition_for_cutlass
from tvm.contrib.cutlass import tune_cutlass_kernels, build_cutlass_kernels
import torchvision
from tvm.relay.transform import ToMixedPrecision
from tvm.contrib.download import download_testdata
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import transform



def convert_conv2d_layout(mod, desired_layouts):
    with tvm.transform.PassContext(opt_level=3):
        seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
        return seq(mod)

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
img = np.expand_dims(img, 0)

batch_size = 16
img = np.tile(img, (batch_size, 1, 1, 1))

sm  = 80
model = torchvision.models.resnet50(pretrained=True).eval()
input_name = "input0"
input_data = torch.from_numpy(img)
scripted_model = torch.jit.trace(model, input_data).eval()

with torch.no_grad():
    torch_res = scripted_model(input_data).numpy()

shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

mod = convert_conv2d_layout(mod, {"nn.conv2d": ["NHWC", "OHWI"]})
mod = ToMixedPrecision("float16")(mod)

with open("models/resnet50.json", "w") as fo:
    fo.write(tvm.ir.save_json(mod))
with open("models/resnet50.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))

