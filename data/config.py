import os
voc = {
    'scale_modules': [
        {'kernel_size': 9, 'stride': 9},
        {'kernel_size': 3, 'stride': 3},
        {'kernel_size': 2, 'stride': 2, 'padding': 1},
        {},
        # {'upscale_factor': 2},
        # {'upscale_factor': 4}
        {'ratio': 2, 'channels': 360, 'size': 9},
        {'ratio': 4, 'channels': 104, 'size': 9}
    ],
    'subnet_modules': {
        'in_channels': [800, 960, 1120, 1280, 360, 104]
    },
    'aspect_ratio': [1.6, 2., 3.],
    'feature_maps': [36, 18, 9, 5, 3, 1],
    'steps': [300//36, 300//18, 300//9, 300//5, 300//3, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'image_size': 300,
    'num_classes': 21,
    'variance': [0.1, 0.2],
    'lr_steps': (500, 600, 700),
    'max_iter': 200000,
    'max_epoch': 900
}

MEANS = (104, 117, 123)


# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))
