<div align="center">
    <h2>Divide and Merge<br/>Motion and Semantic Learning in End-to-End Autonomous Driving
    <br/>
    <br/>
    <a href="https://arxiv.org/abs/2502.07631"><img src='https://img.shields.io/badge/arXiv-Page-aff'></a>
    </h2>
</div>


https://github.com/user-attachments/assets/5d7c65ab-ca9e-47ea-8f66-f1b4998f417d


## Getting Started <a name="usage"></a>
- [Installation](docs/INSTALL.md)
- [Prepare Dataset](docs/DATA_PREP.md)
- [Train/Eval](docs/TRAIN_EVAL.md)
- [Closed-loop Eval](docs/EVAL_NEURONCAP.md)

## News <a name="news"></a>
- **`2025/04/04`** Initial code release
- **`2025/02/11`** DMAD [paper](https://arxiv.org/abs/2502.07631) published on arXiv.

## Todos <a name="todos"></a>
- [ ] SparseDMAD (SparseDrive-based implementation)
- [x] Closed-loop evaluation on NeuroNCAP
- [x] DMAD (UniAD-based implementation)
- [x] Initial code release
- [x] Paper release

## Results <a name="results"></a>

### nuScenes evaluation
| Method<br>stage | Detection<br>NDS | Tracking<br>AMOTA | Mapping<br>IoU-lane | Prediction<br>EPA-car | Planning<br>Avg Col. | Weights |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DMAD<br>stage1 | 0.504 | 0.394 | 0.312 | - | - | [stage1](https://github.com/shenyinzhe/DMAD/releases/download/v1.0/dmad_stage1.pth) |
| DMAD<br>stage2 | 0.506 | 0.393 | 0.321 | 0.535 | 0.127 | [stage2](https://github.com/shenyinzhe/DMAD/releases/download/v1.0/dmad_stage2.pth) |

### NeuroNCAP evaluation

#### NeuroNCAP score:

| Model | Avg | Stationary | Frontal | Side |
| :---: | :---: | :---: | :---: | :---: |
| UniAD | 2.11 | 3.50 | 1.17 | 1.67 |
| DMAD | 2.65 | 4.40 | 1.47 | 2.07 |

#### Collision rate (%)

| Model | Avg  | Stationary | Frontal | Side |
| :---: | :---: | :---: | :---: | :---: |
| UniAD | 60.4 | 32.4 | 77.6 | 71.2 |
| DMAD | 50.1 | 14.8 | 74.0 | 61.6 |

## License <a name="license"></a>

All assets and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.

## Citation <a name="citation"></a>
If you find this work useful, please consider citing:
```bibtex
@article{shen2025divide,
  title={Divide and Merge: Motion and Semantic Learning in End-to-End Autonomous Driving},
  author={Shen, Yinzhe and Ta{\c{s}}, {\"O}mer {\c{S}}ahin and Wang, Kaiwen and Wagner, Royden and Stiller, Christoph},
  journal={arXiv preprint arXiv:2502.07631},
  year={2025}
}
```
## Acknowledgement <a name="acknowledgement"></a>
We appreciate these excellent works:
- [UniAD](https://github.com/OpenDriveLab/UniAD)
- [NeuroNCAP](https://github.com/atonderski/neuro-ncap/)
