from keras_wrapper.dataset import Dataset, saveDataset, loadDataset
from utils.common import get_num_captions
import numpy as np
import copy

import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

def build_dataset(params):
    
    if params['REBUILD_DATASET']: # We build a new dataset instance
        if(params['VERBOSE'] > 0):
            silence=False
            logging.info('Building ' + params['DATASET_NAME'] + ' dataset')
        else:
            silence=True

        base_path = params['DATA_ROOT_PATH']
        name = params['DATASET_NAME']
        ds = Dataset(name, base_path, silence=silence)

        ##### INPUT DATA
        # Let's load the associated images (inputs)
        num_cap = 1 # We only extract one feature vector per image
        list_train = base_path + '/' + params['IMG_FILES']['train'][0]
        list_val = base_path + '/' + params['IMG_FILES']['val'][0]
        list_test = base_path + '/' + params['IMG_FILES']['test'][0]
        ds.setInput(list_train, 'train',
                    type='raw-image', id=params['INPUTS_IDS_DATASET'][0],
                    img_size=params['IMG_SIZE'], img_size_crop=params['IMG_CROP_SIZE'],
                    repeat_set=num_cap)
        ds.setInput(list_val, 'val',
                    type='raw-image', id=params['INPUTS_IDS_DATASET'][0],
                    img_size=params['IMG_SIZE'], img_size_crop=params['IMG_CROP_SIZE'],
                    repeat_set=num_cap)
        ds.setInput(list_test, 'test',
                    type='raw-image', id=params['INPUTS_IDS_DATASET'][0],
                    img_size=params['IMG_SIZE'], img_size_crop=params['IMG_CROP_SIZE'],
                    repeat_set=num_cap)
        ### IMAGES' associated IDs
        ds.setInput(base_path + '/' + params['IMG_FILES']['train'][1], 'train',
                    type='id', id=params['INPUTS_IDS_DATASET'][0] + '_ids',
                    repeat_set=num_cap)
        ds.setInput(base_path + '/' + params['IMG_FILES']['val'][1], 'val',
                    type='id', id=params['INPUTS_IDS_DATASET'][0] + '_ids',
                    repeat_set=num_cap)
        ds.setInput(base_path + '/' + params['IMG_FILES']['test'][1], 'test',
                    type='id', id=params['INPUTS_IDS_DATASET'][0] + '_ids',
                    repeat_set=num_cap)
        # Train mean
        ds.setTrainMean(params['MEAN_IMAGE'], params['INPUTS_IDS_DATASET'][0])

        ###### OUTPUT DATA: None

        # Process dataset for keeping only one caption per image and storing the rest in a dict() with the following format:
        #        ds.extra_variables[set_name][id_output][img_position] = [cap1, cap2, cap3, ..., capN]
        #keep_n_captions(ds, repeat=[1, 1], n=1, set_names=['val','test'])

        # We have finished loading the dataset, now we can store it for using it in the future
        saveDataset(ds, params['DATASET_STORE_PATH'])
    else:
        # We can easily recover it with a single line
        ds = loadDataset(params['DATASET_STORE_PATH']+'/Dataset_'+params['DATASET_NAME']+'.pkl')

    return ds


def keep_n_captions(ds, repeat, n=1, set_names=['val','test']):
    ''' Keeps only n captions per image and stores the rest in dictionaries for a later evaluation
    '''

    for s,r in zip(set_names, repeat):
        logging.info('Keeping '+str(n)+' captions per input on the '+str(s)+' set.')

        ds.extra_variables[s] = dict()
        exec('n_samples = ds.len_'+s)

        # Process inputs
        for id_in in ds.ids_inputs:
            new_X = []
            if id_in in ds.optional_inputs:
                try:
                    exec('X = ds.X_'+s)
                    if isinstance(r, list):
                        i = 0
                        for next_repeat in r:
                            for j in range(n):
                                new_X.append(X[id_in][i+j])
                            i += next_repeat
                    else:
                        for i in range(0, n_samples, r):
                            for j in range(n):
                                new_X.append(X[id_in][i+j])
                    exec('ds.X_'+s+'[id_in] = new_X')
                except: pass
            else:
                exec('X = ds.X_'+s)
                if isinstance(r, list):

                    i = 0
                    for next_repeat in r:
                        for j in range(n):
                            new_X.append(X[id_in][i+j])
                        i += next_repeat
                else:
                    for i in range(0, n_samples, r):
                        for j in range(n):
                            new_X.append(X[id_in][i+j])

                exec('ds.X_'+s+'[id_in] = new_X')
        # Process outputs
        for id_out in ds.ids_outputs:
            new_Y = []
            exec('Y = ds.Y_'+s)
            dict_Y = dict()
            count_samples = 0
            if isinstance(r, list):
                i = 0
                for next_repeat in r:
                    dict_Y[count_samples] = []
                    for j in range(next_repeat):
                        if(j < n):
                            new_Y.append(Y[id_out][i+j])
                        dict_Y[count_samples].append(Y[id_out][i+j])
                    count_samples += 1
                    i += next_repeat
            else:
                for i in range(0, n_samples, r):
                    dict_Y[count_samples] = []
                    for j in range(r):
                        if(j < n):
                            new_Y.append(Y[id_out][i+j])
                        dict_Y[count_samples].append(Y[id_out][i+j])
                    count_samples += 1
            exec('ds.Y_'+s+'[id_out] = new_Y')
            # store dictionary with img_pos -> [cap1, cap2, cap3, ..., capNi]
            ds.extra_variables[s][id_out] = dict_Y
    
        new_len = len(new_Y)
        exec('ds.len_'+s+' = new_len')
        logging.info('Samples reduced to '+str(new_len)+' in '+s+' set.')


def keep_top_N_keywords(ds, N=100, order='desc'):
    '''
        Keeps the top appearing N keywords only.
    '''
    newds = copy.deepcopy(ds)
    
    # check top N keywords w.r.t. training set
    keywords = newds.X_train['query']
    unique_keys = list(set(keywords))
    counts_keys = [[],[]]
    for k in unique_keys:
        counts_keys[0].append(k)
        counts_keys[1].append(keywords.count(k))
    sorted_keys = [i[0] for i in sorted(enumerate(counts_keys[1]), key=lambda x:x[1])]
    
    if order == 'desc':
        sorted_keys = sorted_keys[::-1]
    elif order == 'asc':
        sorted_keys = sorted_keys
    
    N = min(N, len(sorted_keys))
    topN = sorted_keys[:N] # topN identifiers
    topN_keywords = [counts_keys[0][s] for s in topN] # topN keywords
    count_topN = [counts_keys[1][s] for s in topN] # topN keywords' counts
    percentage = float(np.sum(count_topN)) / np.sum(counts_keys[1]) * 100.
    print '\tKeeping top %d/%d keywords (%.2f%%)' % (N, len(sorted_keys), percentage)
    
    
    # Process each data split removing non-desired keywords
    for s in ['train', 'val', 'test']:
        exec('length = newds.len_'+s)
        
        # Only continue if we have data in the current data split
        if length > 0:
            
            # Check keywords we do keep
            to_keep = []
            exec('these_keys = newds.X_'+s+'["query"]')
            for i, k in enumerate(these_keys):
                if k in topN_keywords:
                    to_keep.append(i)
                    
            # Remove remaining keys of inputs (X)
            exec('data_keys = newds.X_'+s+'.keys()')
            for dk in data_keys: 
                exec('this_key_data = newds.X_'+s+'["'+dk+'"]')
                if this_key_data:
                    kept = [this_key_data[i] for i in to_keep]
                    exec('newds.X_'+s+'["'+dk+'"] = kept')
        
            # Remove remaining keys of outputs (Y)
            exec('data_keys = newds.Y_'+s+'.keys()')
            for dk in data_keys:
                exec('this_key_data = newds.Y_'+s+'["'+dk+'"]')
                if this_key_data:
                    kept = [this_key_data[i] for i in to_keep]
                    exec('newds.Y_'+s+'["'+dk+'"] = kept')
        
            # Set new length
            exec('newds.len_'+s+' = len(newds.X_'+s+'["query"])')
            
            # Remove additional evaluation data (if any)
            if s in newds.extra_variables:
                exec('data_keys = newds.extra_variables["'+s+'"].keys()')
                for dk in data_keys:
                    exec('this_key_data = newds.extra_variables["'+s+'"]["'+dk+'"]')
                    if this_key_data:
                        count_keys = 0
                        kept = dict()
                        for i in to_keep:
                            kept[count_keys] = this_key_data[i]
                            count_keys += 1
                        exec('newds.extra_variables["'+s+'"]["'+dk+'"] = kept')
    
    return newds