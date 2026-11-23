<div align="center">
    <h2>Divide and Merge<br/>Motion and Semantic Learning in End-to-End Autonomous Driving
    <br/>
    <br/>
    <a href="https://arxiv.org/abs/2502.07631"><img src='https://img.shields.io/badge/arXiv-Page-aff'></a>
    </h2>
</div>


https://github.com/user-attachments/assets/d02ec267-4f5b-4f77-a3ff-5e38880d67c9


## Getting Started <a name="usage"></a>
- [Installation](docs/INSTALL.md)
- [Prepare Dataset](docs/DATA_PREP.md)
- [Train/Eval](docs/TRAIN_EVAL.md)
- [Closed-loop Eval](docs/EVAL_NEURONCAP.md)
- [KITScenes Multimodal Inference](docs/KITSCENES.md)

## News <a name="news"></a>
- **`2026/11/23`** Zero-shot open-loop inference on [KITScenes Multimodal](https://kitscenes.com/benchmarks/multimodal-e2e-driving/) ([guide](docs/KITSCENES.md))
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

### KITScenes Multimodal evaluation

Zero-shot open-loop results of the nuScenes checkpoints on the 483 samples of the
`val` and `overlap-train-val` scenes of KITScenes Multimodal v2.0.0, from the
[KITScenes leaderboard](https://kitscenes.com/benchmarks/multimodal-e2e-driving/).
ADE/FDE at 3 s; MMS at 5 s. See [KITScenes Inference](docs/KITSCENES.md) to reproduce.

| Model | FDE (m) ↓ | ADE (m) ↓ | Map comp. (%) ↑ | Coll.-free (%) ↑ | CL dist. (m) ↓ | MMS ↑ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UniAD | 4.63 | 2.32 | 61.1 | 89.2 | 0.62 | 3.59 |
| DMAD | 4.47 | 2.28 | 59.4 | 87.6 | 0.63 | 3.62 |

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
