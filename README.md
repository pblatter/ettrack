# E.T.Track - Visual Object Tracking with Exemplar Transformers [WACV22023]
Official implementation of the E.T.Track (WACV22023), including training code and trained models.

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

## Data preparation
* Prepare the data according to the scripts in `crop/` [here](https://github.com/researchmm/TracKit/tree/master/lib/dataset).<br>
* Modify the paths to the training data directory and the annotations json file in `tracking/basic_model/et_tracker.yaml`


## Checkpoints
Download the following checkpoints:
* LightTrack SuperNet checkpoint used for the initialization of the backbone from [here](https://www.filesharing.com/file/details/2806766/model_best.pth.tar). <br>
  Move the checkpoint to `exemplar-transformer-tracking/checkpoints/supernet/`.
* Trained E.T.Track from [here](https://www.filesharing.com/file/details/2806767/checkpoint_e35.pth) <br>
  Move the checkpoint to `exemplar-transformer-tracking/checkpoints/et_tracker/`.

## Training
* Runing the following commands to train the E.T.Tracker.
    ```bash
    conda activate <ENV NAME>
    cd exemplar-transformer-tracking
    bash tracking/basic_model/training.sh
    ```  

## Evaluation
We evaluate our models using [PyTracking](https://github.com/visionml/pytracking).
* Add the correct dataset in `pytracking/experiments/myexperiments.py` (default: OTB-100)
* Run `python3 -m pytracking.run_experiment myexperiments et_tracker --threads 0`
