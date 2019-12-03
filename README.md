# open-images-dataset-loaders

PyTorch dataset classes for the Open Images (v5) dataset.

## Tasks

### Object Detection

Object Detection loader is fully functional.
This is an example of its return tuple:

```python
(
    # The input image (assuming no transforms are passed)
    <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1024x683 at 0x7F3916A75940>,
    # labels list of tuples: (class_id, bbox(xmin, xmax, ymin, ymax))
    [
        (tensor(313), tensor([0.0000, 1.0000, 0.0000, 0.9700])),
        (tensor(110), tensor([0.0000, 1.0000, 0.0000, 0.9625])),
        (tensor(142), tensor([0.0000, 0.9983, 0.0000, 0.9650])),
        (tensor(48), tensor([0.0000, 0.2433, 0.1700, 0.4700])),
        (tensor(48), tensor([0.0000, 0.1167, 0.3475, 0.7125])),
        (tensor(48), tensor([0.0300, 0.1950, 0.6325, 0.9625])),
        (tensor(48), tensor([0.0317, 0.3933, 0.0000, 0.1625])),
        (tensor(48), tensor([0.0933, 0.3917, 0.4650, 0.9350])),
        (tensor(48), tensor([0.1983, 0.6883, 0.0600, 0.3900])),
        (tensor(48), tensor([0.2283, 0.4633, 0.3225, 0.8850])),
        (tensor(48), tensor([0.3850, 0.5200, 0.0000, 0.0450])),
        (tensor(48), tensor([0.4200, 0.6567, 0.5425, 0.8875])),
        (tensor(48), tensor([0.4583, 0.7583, 0.0000, 0.2000])),
        (tensor(48), tensor([0.4817, 0.7417, 0.3825, 0.8475])),
        (tensor(48), tensor([0.5900, 0.9117, 0.3000, 0.7775])),
        (tensor(48), tensor([0.7033, 0.8867, 0.3125, 0.5225])),
        (tensor(48), tensor([0.7317, 0.9983, 0.0250, 0.3125])),
        (tensor(48), tensor([0.8200, 0.9983, 0.2750, 0.8075])),
    ],
)
```

### Object Relationship Detection (ORD)

```python
(
    <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1024x675 at 0x7F00EA904E80>,
    # labels list of tuples: (obj_id_1, obj_id_2, bbox_1, bbox_2, relationship_id))
    [
        (
            tensor(280),
            tensor(602),
            tensor([0.2185, 0.4575, 0.7089, 1.0000]),
            tensor([0.2185, 0.4575, 0.7089, 1.0000]),
            tensor(5),
        ), (
            tensor(96),
            tensor(602),
            tensor([0.5513, 0.8299, 0.8022, 0.9956]),
            tensor([0.5513, 0.8299, 0.8022, 0.9956]),
            tensor(5),
        ), (
            tensor(96),
            tensor(280),
            tensor([0.5499, 0.8299, 0.8000, 0.9956]),
            tensor([0.2185, 0.4589, 0.7067, 1.0000]),
            tensor(0),
        ),
    ],
)
```
