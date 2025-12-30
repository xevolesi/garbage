"""
YuNet ONNX Export Script

Exports the custom YuNet face detector model to ONNX format.
"""

import argparse
import numpy as np
import torch
import onnx

from source.models import YuNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YuNet to ONNX format")
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        required=False,
        default=None,
        help="Path to the checkpoint file (.pt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="yunet.onnx",
        help="Output ONNX file path (default: yunet.onnx)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Input size [height, width] (default: 640 640)",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=11,
        help="ONNX opset version (default: 11)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify ONNX model output against PyTorch",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str | None, device: torch.device) -> YuNet:
    """Load the YuNet model from a checkpoint."""
    model = YuNet(num_classes=1, num_keypoints=5)
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
    model.eval()
    
    return model


def export_onnx(
    model: YuNet,
    output_path: str,
    input_size: tuple[int, int],
    opset_version: int,
) -> None:
    """Export the model to ONNX format."""
    height, width = input_size
    dummy_input = torch.randn(1, 3, height, width)
    
    # Define output names matching the model structure
    # Each stride level outputs: obj, cls, box, kps
    output_names = []
    for stride in [8, 16, 32]:
        output_names.extend([
            f"obj_{stride}",
            f"cls_{stride}",
            f"box_{stride}",
            f"kps_{stride}",
        ])
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=output_names,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )
    
    # Validate the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"Successfully exported ONNX model to: {output_path}")
    print(f"  Input: input [1, 3, {height}, {width}]")
    print(f"  Outputs: {', '.join(output_names)}")


def verify_onnx(
    model: YuNet,
    onnx_path: str,
    input_size: tuple[int, int],
) -> None:
    """Verify ONNX model output matches PyTorch output."""
    import onnxruntime as ort
    
    height, width = input_size
    test_input = torch.randn(1, 3, height, width)
    
    # PyTorch inference
    model.eval()
    with torch.no_grad():
        pytorch_outputs = model(test_input)
    
    # Flatten PyTorch outputs: ((obj8, cls8, box8, kps8), (obj16, ...), (obj32, ...))
    pytorch_flat = []
    for level_outputs in pytorch_outputs:
        for tensor in level_outputs:
            pytorch_flat.append(tensor.numpy())
    
    # ONNX Runtime inference
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {"input": test_input.numpy()})
    
    # Compare outputs
    print("\nVerification Results:")
    all_close = True
    for i, (pt_out, onnx_out) in enumerate(zip(pytorch_flat, onnx_outputs)):
        try:
            np.testing.assert_allclose(pt_out, onnx_out, rtol=1e-4, atol=1e-5)
            print(f"  Output {i}: ✓ Match")
        except AssertionError as e:
            print(f"  Output {i}: ✗ Mismatch - {e}")
            all_close = False
    
    if all_close:
        print("\n✓ All outputs match between PyTorch and ONNX!")
    else:
        print("\n⚠ Some outputs differ (this may be acceptable due to floating-point precision)")


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    input_size = tuple(args.input_size)
    print(f"Exporting with input size: {input_size}")
    
    export_onnx(model, args.output, input_size, args.opset_version)
    
    if args.verify:
        verify_onnx(model, args.output, input_size)


if __name__ == "__main__":
    main()

