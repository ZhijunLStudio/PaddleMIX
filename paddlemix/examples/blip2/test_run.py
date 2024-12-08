import os
# 获取项目的根目录（`paddlemix` 的父目录）
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from paddlemix.models.blip2.configuration import Blip2Config
from paddlemix.models.blip2.modeling import Blip2ForConditionalGeneration


name = "paddlemix/blip2-caption-opt2.7b"

blip2_config = Blip2Config.from_pretrained(name)
model = Blip2ForConditionalGeneration(blip2_config)