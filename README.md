## Utilizing CFPT neck for YOLO11

### 🚧WORK IN PROGRESS

This project modifies the **[Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)** architecture by replacing its default neck (PAFPN) with the **[Cross-layer Feature Pyramid Transformer (CFPT)](https://github.com/duzw9311/CFPT)** to improve small object detection in sparse LiDAR BEV images.

## Howto(Colab recommended)
```python
# 1. Install ultralytics
!pip install ultralytics
```
```python
# 2. Clone Repository
!git clone https://github.com/Yongha026/YOLO11-CFPT.git
```
```python
# 3. Replace library
import shutil

shutil.rmtree("/usr/local/lib/python3.12/dist-packages/ultralytics/nn/")
shutil.copytree("/content/YOLO11-CFPT/ultralytics/ultralytics/nn/","/usr/local/lib/python3.12/dist-packages/ultralytics/nn")
```
```python
# 4. Inference
from ultralytics import YOLO

model = YOLO("/content/YOLO11-CFPT/ultralytics/ultralytics/cfg/models/11/yolo11n-CFPT-obb.yaml", task='obb')
```
