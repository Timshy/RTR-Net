# Training

RTR-Net is trained with RGB images, category response maps, foreground masks, and skeletons derived from the masks.

```bash
python tools/train_rtrnet.py --config configs/config_rtrnet.yaml
```

The open-vocabulary model is frozen and is only used to generate response maps.
