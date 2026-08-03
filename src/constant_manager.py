class ColumnNames:
    """
    Class to hold column names for the dataset.
    """
    # Feature information for raw data
    RAW_INPUT_COLUMNS = ['Engine_speed', 'Boost pressure', 'Mass1', 'Mass2', 'SOI1', 'SOI2', 'IVO', 'IVC', 'EVO', 'EVC']
    RAW_REDUCED_INPUT_COLUMNS = ['Engine_speed', 'Boost pressure', 'Mass1', 'Mass2', 'SOI2', 'IVO', 'IVC', 'EVO', 'EVC']
    RAW_REDUCED_INPUT_COLUMNS_ENG = ['Engine_speed', 'Boost pressure', 'Mass1', 'Mass2', 'SOI2', 'IVO', 'IVC', 'EVO', 'EVC', 'Reactivity_Ratio']
    RAW_OUTPUT_COLUMNS = ['IEMP', 'ITE', 'CA50', 'Lambda', 'PRR4_max', 'Pmax', 'Nox', 'CH4', 'CO', 'NMHC', 'CO2']
    RAW_CATEGORICAL_COLUMNS = []

    # Reader information
    READER_TYPE_XLSX = 'xlsx'
    READER_TYPE_PARQUET = 'parquet'
    ALLOWED_READER_TYPES = {READER_TYPE_XLSX, READER_TYPE_PARQUET}

class BoundaryConstants:
    """
    Class to hold constants related to boundary estimation.
    """
    UNSCALED_BOUNDS = {
        'Engine_speed': [1000.00, 1000.00], 
        'Boost pressure': [1.00, 4.00],
        'Mass1': [2.50, 35.00],
        'Mass2': [0.00, 5.00],
        'SOI1': [290.00, 290.00],
        'SOI2': [30.00, 100.00],
        'IVO': [340.00, 460.00],
        'IVC': [480.00, 600.00],
        'EVO': [120.00, 220.00],
        'EVC': [260.00, 370.00]
    }


class NoiseConstants:
    """
    Class to hold constants related to noise estimation.
    """
    EMPIRICAL_METHOD = 'empirical'
    MLE_METHOD = 'mle'
    MLE_BASIC_METHOD = 'mle_basic'
    STANDARD_NOISE_METHOD = 'standard'
    RESIDUAL_METHOD = 'residual'
    NOISE_LEVEL_BOUNDS = (1e-5, 1e50)
    DEFAULT_NOISE_LEVEL = 0.01
    RANDOM_STATE = 42
    
class PredictConstants:
    """
    Class to hold constants related to predictions.
    """
    DEFAULT_N_TEST_SAMPLES = 500
    DEFAULT_SCALING = False
    METHOD_TEST_TRAIN_SPLIT = 'test_train_split'
    METHOD_SINGLE_SAMPLE = 'single_sample'
    METHOD_ONLY_ONE_SAMPLE = 'only_one_sample'

class ConstantManager:
    def __init__(self):
        for cls in [ColumnNames, BoundaryConstants, NoiseConstants, PredictConstants]:
            for key, value in cls.__dict__.items():
                if not key.startswith("__"):
                    self.__dict__.update(**{key: value})
                    
    def __setattr__(self, name, value):
        raise TypeError("Constants are immutable")