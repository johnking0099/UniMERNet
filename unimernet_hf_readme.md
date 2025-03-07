# unimernet_hf demo

unimernet_hf is a huggingface style api for unimernet.

Here are some small demos.

## Load model

For huggingface's `save_pretrained()` model, use following code.

```python
from unimernet_hf import UnimernetModel

model = UnimernetModel.from_pretrained("/path/to/your/model/dir")
```

For the old model created by `torch.save()`, use following code.

```python
from unimernet_hf import UnimernetModel

model = UnimernetModel.from_checkpoint("/path/to/your/old_model/dir")
```

## Generate

Use following code to generate in batch.

```python
dataset = ImageDataset(images, transform=model.transform)
dataloader = DataLoader(dataset, batch_size=256, num_workers=1)
output_strings = []

for batch in dataloader:
    batch = batch.to(dtype=model.dtype)
    batch = batch.to(model.device)
    
    output = model.generate({'image': batch})
    output_strings.extend(output['fixed_str'])
```

## About SDPA

To use torch's SDPA operator, Unimernet's base Model, VisionEncoderDecoderModel should support SDPA.
The VisionEncoderDecoderModel starts to support SDPA from `transformers-4.46.0` .
So, you must have `transformers>=4.46.0` installed to enable SDPA for UnimernetModel.

