from pytracking.utils import TrackerParams
from tracking.basic_model.et_tracker import ET_Tracker

def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.use_gpu = False

    params.checkpoint_epoch = 35

    params.net = ET_Tracker(search_size=256,
                            template_size=128,
                            stride=16,
                            e_exemplars=4,
                            sm_normalization=True, 
                            temperature=2,
                            dropout=False)

    params.big_sz = 288
    params.small_sz = 256
    params.stride = 16
    params.even = 0
    params.model_name = 'et_tracker'

    params.image_sample_size = 256
    params.image_template_size = 128
    params.search_area_scale = 5

    params.window_influence = 0
    params.lr = 0.616
    params.penalty_k = 0.007
    params.context_amount = 0.5

    params.features_initialized = False

    return params