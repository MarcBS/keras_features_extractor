def load_parameters():
    """
        Loads the defined parameters
    """
    # Input data params
    DATA_ROOT_PATH = '/media/HDD_2TB/DATASETS/MSCOCO/'

    INPUT_DATA_TYPE = 'raw-image'  # 'image-features' or 'raw-image' [Optional: '-query']
    DATASET_NAME = 'MSCOCO_' + INPUT_DATA_TYPE  # Dataset name

    MEAN_IMAGE = [103.939, 116.779, 123.68]  # Training mean image values for each channel (RGB)
    #IMG_SIZE = [256, 256, 3]  # Size of the input images (will be resized to the desired size)
    #IMG_CROP_SIZE = [224, 224, 3] # Size of the image crops inputted to the model

    # InceptionV3
    IMG_SIZE = [342, 342, 3]  # Size of the input images (will be resized to the desired size)
    IMG_CROP_SIZE = [299, 299, 3]  # Size of the image crops inputted to the model

    # Image and features files (the chars {} will be replaced by each type of features)
    IMG_FILES = {'train': ['Annotations/train_list_images.txt',
                           'Annotations/train_list_ids.txt'],
                 'val': ['Annotations/val_list_images.txt',
                         'Annotations/val_list_ids.txt'],
                 'test': ['Annotations/test_list_images.txt',
                          'Annotations/test_list_ids.txt']
                 }
    FEATURE_NAMES = []

    # Prepare input mapping between dataset and model
    INPUTS_IDS_DATASET = ['image']  # Corresponding inputs of the dataset
    INPUTS_IDS_MODEL = ['input_1']  # Corresponding inputs of the built model ('input_1' for ResNet50)

    # Prepare output mapping between dataset and model
    OUTPUTS_IDS_DATASET = ['mixed10']  # Corresponding outputs of the dataset
    OUTPUTS_IDS_MODEL = ['mixed10']  # Corresponding outputs of the built model


    # Evaluation params
    METRICS = []                                  # Metric used for evaluating model after each epoch
                                                  # (leave empty if only prediction is required)
    EVAL_ON_SETS = ['train', 'val', 'test']       # Possible values: 'train', 'val' and 'test' (external evaluator)
    EVAL_ON_SETS_KERAS = []                       # Possible values: 'train', 'val' and 'test' (Keras' evaluator)
    START_EVAL_ON_EPOCH = 0                       # First epoch where the model will be evaluated
    EVAL_EACH_EPOCHS = True                       # Select whether evaluate between N epochs or N updates
    EVAL_EACH = 0                                 # Number of epochs/updates between each evaluation

    # Input data parameters
    DATA_AUGMENTATION = True                      # Apply data augmentation on input data
                                                  # (noise on features, random crop on images)
    SHUFFLE_TRAIN = False                          # Apply shuffling on training data at the beginnin of each epoch

    CLASSIFIER_ACTIVATION = 'softmax'

    # Optimizer parameters (see model.compile() function)
    LOSS = 'categorical_crossentropy'
    LR_DECAY = None  # number of minimum number of epochs before the next LR decay (set to None for disabling)
    LR_GAMMA = 0.99  # multiplier used for decreasing the LR

    OPTIMIZER = 'Adam'      # Optimizer
    LR = 0.001              # (recommended values - Adam 0.001 - Adadelta 1.0
    PRE_TRAINED_LR_MULTIPLIER = 0.0001   # LR multiplier assigned to pre-trained network (0.001 recommended)
    PRE_TRAINED_LEARNABLE = False # finetune pre-trained network?
    WEIGHT_DECAY = 1e-4     # L2 regularization
    CLIP_C = 10.            # During training, clip gradients to this norm
    SAMPLE_WEIGHTS = True   # Select whether we use a weights matrix (mask) for the data outputs

    # Training parameters
    BATCH_SIZE = 32
    PARALLEL_LOADERS = 8        # Parallel data batch loaders
    WRITE_VALID_SAMPLES = True  # Write valid samples in file
    PREDICTION_STEP = 100      # Load this number of samples into memory when predicting

    # Possible MODEL_TYPE values:
    #                          [ Available Models List ]
    #
    #                          InceptionV3
    # ===
    MODEL_TYPE = 'InceptionV3'

    # Results plot and models storing parameters
    EXTRA_NAME = '' # This will be appended to the end of the model name
    MODEL_NAME = DATASET_NAME + '_' + MODEL_TYPE + '_'.join(FEATURE_NAMES) + '_' + OUTPUTS_IDS_MODEL[0]

    MODEL_NAME += EXTRA_NAME

    STORE_PATH = 'trained_models/' + MODEL_NAME  + '/' # Models and evaluation results will be stored here
    DATASET_STORE_PATH = 'datasets/'                   # Dataset instance will be stored here

    SAMPLING_SAVE_MODE = 'npy'                         # 'list', 'npy', 'hdf5'
    VERBOSE = 1                                        # Verbosity level
    RELOAD = 0                                         # If 0 start training from scratch, otherwise the model
                                                       # Saved on epoch 'RELOAD' will be used
    REBUILD_DATASET = True                             # Build again or use stored instance
    MODE = 'sampling'                                  # 'training' or 'sampling' (if 'sampling' then RELOAD must
                                                       # be greater than 0 and EVAL_ON_SETS will be used)

    # Extra parameters for special trainings
    TRAIN_ON_TRAINVAL = False  # train the model on both training and validation sets combined
    FORCE_RELOAD_VOCABULARY = False  # force building a new vocabulary from the training samples applicable if RELOAD > 1

    # ============================================
    parameters = locals().copy()
    return parameters
