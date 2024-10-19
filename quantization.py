import torch

def quantize_qint8(net):
    example_inputs = torch.randn(1, 1, 32, 32)
    model_dynamic_quantized = torch.quantization.quantize_dynamic(
        net.to('cpu'), qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
    ).to('cpu')


    return model_dynamic_quantized
