# STDN: Scale-Transferrable Object Detection
A [PyTorch](http://pytorch.org/) implementation of [Scale-Transferrable Object Detection](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1376.pdf)

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
  