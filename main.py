import sys
for p in sys.path:
    if 'keras_marc-1.0.10' in p:
        sys.path.remove(p)

import logging
import time

from config import load_parameters
from data_engine.prepare_data import build_dataset
from feature_extractor import Feature_Extractor

from keras_wrapper.cnn_model import loadModel, saveModel
from keras_wrapper.extra.callbacks import EvalPerformance, SampleEachNUpdates
from keras_wrapper.extra.evaluation import select as selectMetric
from keras_wrapper.extra.read_write import dict2file, list2file, numpy2file, numpy2hdf5

import sys
import ast
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def train_model(params):
    """
        Main function
    """
    raise BaseException, 'Training is not available when extracting features'

def apply_Feature_Extractor_model(params, dataset=None, extractor_model=None):
    """
        Function for using a previously trained model for sampling.
    """

    ########### Load data
    if dataset is None:
        dataset = build_dataset(params)

    ########### Load model
    if extractor_model is None and params['RELOAD'] > 0:
        extractor_model = loadModel(params['STORE_PATH'], params['RELOAD'])
    else:
        extractor_model = Feature_Extractor(params,
                                            type=params['MODEL_TYPE'],
                                            verbose=params['VERBOSE'],
                                            model_name=params['MODEL_NAME'],
                                            store_path=params['STORE_PATH'])
        # Define the inputs and outputs mapping from our Dataset instance to our model
        inputMapping = dict()
        for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
            if len(extractor_model.ids_inputs) > i:
                pos_source = dataset.ids_inputs.index(id_in)
                id_dest = extractor_model.ids_inputs[i]
                inputMapping[id_dest] = pos_source
        extractor_model.setInputsMapping(inputMapping)

    ########### Apply sampling
    extra_vars = dict()
    for s in params["EVAL_ON_SETS"]:
        # Apply model predictions
        params_prediction = {'batch_size': params['BATCH_SIZE'],
                             'n_parallel_loaders': params['PARALLEL_LOADERS'],
                             'predict_on_sets': [s],
                             'verbose': 0}
        logging.info("<<< Predicting outputs of " + s + " set >>>")

        if params['SAMPLING_SAVE_MODE'] == 'list':
            filepath = extractor_model.model_path + '/' + s + '_sampling.pred' # results file
            list2file(filepath, [], permission='w')

        start_time = time.time()
        eta = -1
        mode = 'w'
        for n_sample in range(0, eval('dataset.len_' + s), params.get('PREDICTION_STEP', 100)):
            params_prediction['init_sample'] = n_sample
            params_prediction['final_sample'] = min(n_sample + params.get('PREDICTION_STEP', 100), eval('dataset.len_' + s))
            predictions = extractor_model.predictNet(dataset, params_prediction)[s]
            # Store result
            if params['SAMPLING_SAVE_MODE'] == 'list':
                filepath = extractor_model.model_path + '/' + s + '_sampling.pred' # results file
                list2file(filepath, predictions, permission='a')
            elif params['SAMPLING_SAVE_MODE'] == 'npy':
                filepath = extractor_model.model_path + '/' + s + '_' + params.get('MODEL_TYPE', '') + '_features.npy'
                numpy2file(filepath, predictions, permission=mode)
            elif params['SAMPLING_SAVE_MODE'] == 'hdf5':
                filepath = extractor_model.model_path + '/' + s + '_' + params.get('MODEL_TYPE', '') + '_features.hdf5'
                numpy2hdf5(filepath, predictions, permission=mode)
            else:
                raise Exception, 'Only "list" or "hdf5" are allowed in "SAMPLING_SAVE_MODE"'
            mode = 'a'
            sys.stdout.write('\r')
            sys.stdout.write("\t Processed %d/%d  -  ETA: %ds " % (n_sample, eval('dataset.len_' + s), int(eta)))
            sys.stdout.flush()
            eta = (eval('dataset.len_' + s) - n_sample) * (time.time() - start_time) / max(n_sample, 1)

def check_params(params):
    if params.get('METRICS') is not None:
        logging.warn("In the feature extraction mode, we won't compute any metric")

    return True

if __name__ == "__main__":

    params = load_parameters()

    try:
        for arg in sys.argv[1:]:
            k, v = arg.split('=')
            params[k] = ast.literal_eval(v)
    except:
        print 'Overwritten arguments must have the form key=Value'
        exit(1)

    if params['MODE'] == 'training':
        logging.info('Running training.')
        train_model(params)
    elif params['MODE'] == 'sampling' or params['MODE'] == 'feature_extraction':
        logging.info('Extracting features.')
        apply_Feature_Extractor_model(params)

    logging.info('Done!')
