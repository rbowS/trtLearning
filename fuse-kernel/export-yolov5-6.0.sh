#!/bin/bash

cd yolov5-copy
python export.py --weights=yolov5s.pt --dynamic --include=onnx --opset=11

mv yolov5s.onnx ../workspace/yolov5s.onnx