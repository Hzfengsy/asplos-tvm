from tvm import relay
from tvm.relay import op
from tvm.relay.dataflow_pattern import *


def reshape_gelu_pattern(inp, bias, inv_sqrt):
    reshape = is_op("reshape")(inp)
    add = is_op("add")(reshape, bias) | is_op("nn.bias_add")(reshape, bias)
    mul = is_op("multiply")(add, inv_sqrt)
    cast_fp32 = is_op("cast")(mul)
    erf = is_op("erf")(cast_fp32)
    mul = is_op("multiply")(erf, is_constant())
    add_cast_fp32 = is_op("cast")(add)
    mul_add_half = is_op("add")(is_constant(), mul)
    mul_fp32 = is_op("multiply")(add_cast_fp32, mul_add_half)
    reshape = is_op("reshape")(mul_fp32)
    return is_op("cast")(reshape)


def convert_reshape_gelu(inp, bias, inv_sqrt):
    bias_out = inp + bias
    mul = bias_out * inv_sqrt
    erf = op.cast(op.erf(op.cast(mul, "float32")), "float16")
    mul_half = erf * relay.const(0.5, dtype="float16")
    return (mul_half + relay.const(0.5, dtype="float16")) * bias_out


class ReshapeGeLURewrite(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.inp = wildcard()
        self.bias = wildcard()
        self.inv_sqrt = wildcard()
        self.pattern = reshape_gelu_pattern(self.inp, self.bias, self.inv_sqrt)

    def callback(self, pre, post, node_map):
        inp = node_map[self.inp][0]
        bias = node_map[self.bias][0]
        inv_sqrt = node_map[self.inv_sqrt][0]
        return convert_reshape_gelu(inp, bias, inv_sqrt)


def rewrite_reshape_gelu(mod):
    mod["main"] = rewrite(ReshapeGeLURewrite(), mod["main"])
    return mod


def convert_conv2d_layout(mod, desired_layouts):
    with tvm.transform.PassContext(opt_level=3):
        seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
        return seq(mod)


def check_params_tensorcore_compatible(prim_func):
    params = prim_func.params
    buffer_map = prim_func.buffer_map
    buffers = [buffer_map[param] for param in params[:2]]
    for buffer in buffers:
        if buffer.shape[-1] % 8 != 0 or buffer.shape[-2] % 8 != 0:
            return False
    return True


def check_params_conv2d_tensorcore_compatible(prim_func):
    params = prim_func.params
    buffer_map = prim_func.buffer_map
    buffers = [buffer_map[param] for param in params[:2]]
    X = buffers[0]
    Weight = buffers[1]
    N, H, W, C = X.shape
    O, S, C, K = Weight.shape
    return K >= 3
    if (N * H * W) % 16 == 0 and (C * K * S) % 16 == 0 and O % 16 == 0:
        return True
    return False


def should_use_memhammer(task):
    mod = task.dispatched[0]
    global_var = mod.get_global_vars()[0]
    task_name = global_var.name_hint
    if "dense" in task_name or "batch_matmul" in task_name:
        # return True
        prim_func = mod[global_var]
        return check_params_tensorcore_compatible(prim_func)
    elif "conv" in task_name:
        prim_func = mod[global_var]
        return check_params_conv2d_tensorcore_compatible(prim_func)
    else:
        return False
