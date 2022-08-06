import argparse
import torch
import yaml
import numpy as np
import torch
from lib.models.model import Model


def export_onnx(model, dummy_input, opt):
    import onnx

    if opt.dynamic:
        torch.onnx.export(model, dummy_input, opt.output, 
                         verbose=False, 
                         opset_version=opt.opset,
                         do_constant_folding=True,
                         input_names=['input'],
                         output_names=['output'],
                         dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})
    else:
        torch.onnx.export(model, dummy_input, opt.output, 
                         verbose=False, 
                         opset_version=opt.opset,
                         do_constant_folding=True,
                         input_names=['input'],
                         output_names=['output'])

    print("Exported to %s!" % opt.output)

    # Checks
    print("Testing onnx model...")

    model_onnx = onnx.load(opt.output)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    # print(onnx.helper.printable_graph(model_onnx.graph))  # print
    
    import onnxruntime
    ort_session = onnxruntime.InferenceSession(opt.output)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # computes ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compares ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(model(dummy_input)), ort_outs[0], rtol=1e-03, atol=1e-05)


def main(opt, cfg):
    if not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(opt.device)
    if not torch.cuda.is_available():
        device = 'cpu'
    model = Model(cfg)
    weights_path = opt.weights
    if not weights_path:
        raise Exception("Please specify path to model weights in config file!")
    weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights['state_dict'])
    model.to(device)
    model.eval()
        
    dummy_input = torch.zeros(opt.batch, cfg["MODEL"]["INPUT_SHAPE"][0], 
                                cfg["MODEL"]["INPUT_SHAPE"][1]).to(device)
    if opt.format == "onnx":
        export_onnx(model, dummy_input, opt)
    else:
        raise Exception("%s format is not supported!" % opt.format)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, 
                        help='path to model weights')
    parser.add_argument('--cfg', type=str, required=True, 
                        help='path to config file')
    parser.add_argument('--format', type=str, default="onnx", 
                        help='format to export')
    parser.add_argument('--output', type=str, required=True, 
                        help='output file path to export')
    parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', 
                        help='dynamic axes')
    parser.add_argument('--opset', type=int, default=12, 
                        help='ONNX: opset version')
    opt = parser.parse_args()
    
    with open(opt.cfg, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()
            
    main(opt, cfg)