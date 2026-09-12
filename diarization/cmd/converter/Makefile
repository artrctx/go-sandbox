run:
	python ./converter.py --checkpoint ./segmentation/pytorch_model.bin --output ../../models/segmentation.onnx

install:
	pip install torch numpy onnx onnxruntime Cython packaging && pip install nemo_toolkit[asr] 
