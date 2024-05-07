import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from zero123_nodes import Zero123, Zero123Preprocess

sys.path.pop(0)

NODE_CLASS_MAPPINGS = {
    "Zero123: Image Rotate in 3D" : Zero123,
    "Zero123: Image Preprocess" : Zero123Preprocess
}