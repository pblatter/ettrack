# E.T.Track - Efficient Visual Tracking with Exemplar Transformers
Official implementation of [E.T.Track](https://arxiv.org/abs/2112.09686). 
E.T.Track utilized our proposed Exemplar Transformer, a transformer module utilizing a single instance level attention layer for realtime visual object tracking.
E.T.Track is up to 8x faster than other transformer-based models, and consistently outperforms competing lightweight trackers that can operate in realtime on standard CPUs. 


E.T.Track        |  The [standard attention](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) vs our Exemplar Attention on the right 
:----------------------------------------------------------:|:----------------------------------------------------------:
<img src='assets/ET.png' align="center" height=400>   |  <img src='assets/V2_att_module.png' align="center" height=300>

## Installation

#### Install dependencies

Install the python environment using the environment file `ettrack_env.yml`.

Generate the relevant files:
```
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```

* Modify local.py.
    Modify the path files for the evaluation in `pytracking/evaluation/local.py`


## Download checkpoints


* Trained E.T.Track model for inference:
```
wget https://data.vision.ee.ethz.ch/kanakism/checkpoint_e35.pth -P ./checkpoints/et_tracker/ 
```

## Evaluation
We evaluate our models using [PyTracking](https://github.com/visionml/pytracking).
The sequences comparing E.T.Track and LT-Mobile in the ''Video Visualizations'' section can be found [here](https://youtu.be/pkiWST8mRuU).
* Add the correct dataset in `pytracking/experiments/myexperiments.py` (default: OTB-100)
* Run `python3 -m pytracking.run_experiment myexperiments et_tracker --threads 0`

## Citation

If you use this code, please consider citing the following paper:

```
@inproceedings{blatter2023efficient,
  title={Efficient visual tracking with exemplar transformers},
  author={Blatter, Philippe and Kanakis, Menelaos and Danelljan, Martin and Van Gool, Luc},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1571--1581},
  year={2023}
}
```