# Point2Skeleton

This repository contains the source code for the CVPR 2021 oral paper [Point2Skeleton: Learning Skeletal Representations from Point Clouds](https://arxiv.org/abs/2012.00230), where we introduce an unsupervised method to generate skeletal meshes from point clouds. 

<a>
    <img src="doc/point2skeleton.jpg" width="70% height="70%"/>
</a>


## Skeletal Mesh

                    
We introduce a generalized skeletal representation, called skeletal mesh. Several good properties of the skeletal mesh make it a useful representation for shape analysis:

- **Recoverability** The skeletal mesh can be considered as a complete shape descriptor, which means it can reconstruct the shape of the original domain. 

- **Abstraction** The skeletal mesh captures the fundamental geometry of a 3D shape and extracts its global topology; the tubular parts are abstracted by simple 1D curve segments and the planar or bulky parts by 2D surface triangles. 

- **Structure awareness** The 1D curve segments and 2D surface sheets as well as the non-manifold branches on the skeletal mesh give a structural differentiation of a shape.

- **Volume-based closure** The interpolation of the skeletal spheres gives solid cone-like or slab-like primitives; then a local geometry is represented by volumetric parts, which provides better integrity of shape context. The interpolation also forms a closed watertight surface.

<a>
    <img src="doc/skeletal_mesh.jpg" width="70% height="70%"/>
</a>


## Code
### Installation
This repository was tested under Python 3.13, PyTorch 2.7.0 (CUDA 12.8), NumPy 2.4.4 on Ubuntu 22.04.
You need to install the dependencies via UV. Please install UV and run `uv sync` to install annd build all modules.

```
apt-get update && apt-get install curl
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

If `uv sync` fails to build PointNet++, please run the following make command:
```
make sync
```

### TensorboardX
To launch tensorboard from terminal, run the following command:
```
tensorboard --logdir tensorboard
```

### Training
* Example command with required parameters:
```
cd src
python train.py --pc_list_file ../data/data-split/all-train.txt --data_root ../data/pointclouds/ --point_num 2000 --skelpoint_num 100 --gpu 0
``` 
* Can simply call `python train.py` once the data folder `data/` is prepared.
* See `python train.py --help` for all the training options. Can change the setting by modifying the parameters in `src/config.py`

### Testing
* Example command with required parameters:
```
cd src
python test.py --pc_list_file ../data/data-split/all-test.txt --data_root ../data/pointclouds/ --point_num 2000 --skelpoint_num 100 --gpu 0 --load_skelnet_path ../weights/weights-skelpoint.pth --load_gae_path ../weights/weights-gae.pth --save_result_path ../results/
``` 
* Can also simply call `python test.py` once the data folder `data/` and network weight folder `weights/` are prepared.
* See `python test.py --help` for all the testing options. 

### Download 
* Train/test data [data.zip](https://1drv.ms/u/s!AlrKELj1ZvAndafaO-Vic70KIus?e=uq53tW).
* Pre-trained model [weights.zip](https://1drv.ms/u/s!AlrKELj1ZvAnd5pJTH48YhJXBf8?e=tPAqG7).
* Unzip the downloaded files to replace the `data/` and `weights/` folders; then you can run the code by simply calling `python train.py` and `python test.py`.
* Dense point cloud [data_dense.zip](https://1drv.ms/u/s!AlrKELj1ZvAneSIupp5qJDE5pJE?e=oWsdFv) and simplified MAT [MAT.zip](https://1drv.ms/u/s!AlrKELj1ZvAndu8fXM5RF64tX6M?e=3v7UM3) for evaluation.

## Acknowledgement
We would like to acknowledge the following projects:

[Unsupervised Learning of Intrinsic Structural Representation Points](https://github.com/NolenChen/3DStructurePoints)

[Pointnet++ PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)

[Graph Convolution](https://github.com/linhaojia13/GCN_pointcloud)

## Citation
If you find our work useful in your research, please consider citing:

```
@InProceedings{Lin_2021_CVPR,
    author    = {Lin, Cheng and Li, Changjian and Liu, Yuan and Chen, Nenglun and Choi, Yi-King and Wang, Wenping},
    title     = {Point2Skeleton: Learning Skeletal Representations from Point Clouds},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {4277-4286}
}
```

## Contact
If you have any questions, please email [Cheng Lin](https://clinplayer.github.io/) at chlin@connect.hku.hk.
