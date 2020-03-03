# open-images-dataset-loaders

PyTorch dataset classes for the Open Images (v5) dataset.

## Tasks

### Object Detection

Object Detection loader is probably functional.
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
        ...
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
