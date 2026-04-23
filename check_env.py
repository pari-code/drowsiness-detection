import sys

def check(name, fn):
    try:
        result = fn()
        print(f"  OK  {name}: {result}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")

print("--- Environment Check ---")
print(f"Python: {sys.version}")

check("torch",        lambda: __import__('torch').__version__)
check("torchvision", lambda: __import__('torchvision').__version__)
check("CUDA",         lambda: str(__import__('torch').cuda.is_available()))
check("cv2",          lambda: __import__('cv2').__version__)
check("mediapipe",   lambda: __import__('mediapipe').__version__)
check("PIL",          lambda: __import__('PIL').__version__)
check("sklearn",     lambda: __import__('sklearn').__version__)
check("matplotlib",  lambda: __import__('matplotlib').__version__)
check("wandb",       lambda: __import__('wandb').__version__)
check("fastapi",     lambda: __import__('fastapi').__version__)
check("onnx",        lambda: __import__('onnx').__version__)

# Quick tensor test
import torch
t = torch.randn(3, 224, 224)
print(f"  OK  tensor shape test: {t.shape}")
print("--- Done. Fix any FAIL lines before proceeding. ---")