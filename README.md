# HFUSOD

Source code of our work: "HFUSOD:Hierarchical fusion of Swin Transformer with CNN network for underwater salient object detection".   
In this works, we propose a new underwater slient object detection algorithm.

## Datasets
Our USOD dataset include the USOD10K, USOD, and the UFO-120 underwater dataset.

## Properties
1. **A unified interface for new models.** To develop a new model, you only need to 1) set configs; 2) define network; 3) define loss function. See methods/template.
2. Setting different backbones through ```--backbone```. **(Available backbones: ResNet-50, VGG-16, MobileNet-v2, EfficientNet-B0, GhostNet, Res2Net)**
3. **Testing all models on your own device.** You can test all available methods in our benchmark, including FPS, MACs, model size and multiple effectiveness metrics.
4. We implement a **loss factory** that you can change the loss functions through ```--loss``` and ```--lw```. 

Thanks for citing our work
```xml
@article{hfusod,
  title={HFUSOD:Hierarchical fusion of Swin Transformer with CNN network for underwater salient object detection},
  author={Weiliang huang, Daqi Zhu},
  journal={IET Image Processing},
  year={2025}
}
```
