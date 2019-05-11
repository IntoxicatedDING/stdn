# STDN: Scale-Transferrable Object Detection
A [PyTorch](http://pytorch.org/) implementation of [Scale-Transferrable Object Detection](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1376.pdf) by Peng Zhou, Bingbing Ni, Cong Geng, Jianguo Hu and Yi Xu.

## Training
To train a network:

```Shell
python train.py
```

- Important options:
  * `dataset_root`: Dataset root directory path
  * `resume`: Checkpoint state_dict file to resume training from.

## Evaluation
To evaluate a trained network:

```Shell
python eval.py
```

- Important options:
  * `trained_model`: Trained state_dict file path to open
  * `voc_root`: Location of VOC root directory
  
## Declaration

The main framework is ported from [SSD in PyTorch](https://github.com/amdegroot/ssd.pytorch).
  
## References
  
- Peng Zhou, Bingbing Ni, Cong Geng, Jianguo Hu, Yi Xu; The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 528-537

- [SSD in PyTorch](https://github.com/amdegroot/ssd.pytorch)
