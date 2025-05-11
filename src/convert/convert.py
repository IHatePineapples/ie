#!/bin/env python3
import torch
from sys import argv

from model import ImageClassifierModel as Model


def new(f: str) -> str:
    f = f.rsplit(sep=".", maxsplit=1)[0]
    return f + ".onnx"


if __name__ == "__main__":
    filename = argv[1]
    model = torch.jit.load(filename)
    example_input = (torch.randn(1, 1, 32, 32),)
    onnx_program = torch.onnx.export(model, example_input)
    onnx_program.optimize()
    onnx_program.save(new(filename))
