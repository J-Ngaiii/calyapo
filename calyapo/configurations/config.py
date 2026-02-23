from pathlib import Path

UNIVERSAL_FINAL_FOLDER = Path('calyapo/data/final')
UNIVERSAL_PENULTIMATE_FOLDER = Path('calyapo/data/penultimate')
UNIVERSAL_RANDOM_SEED = 42
UNIVERSAL_NA_FILLER = "not available"
DATA_PATHS = {
    'IGS' : {
        'raw' : Path('calyapo/data/raw/igs'), 
        'intermediate' : Path('calyapo/data/intermediate/igs'), 
        'processed' : Path('calyapo/data/processed/igs'), 
        'penultimate' : UNIVERSAL_PENULTIMATE_FOLDER, 
        'final' : UNIVERSAL_FINAL_FOLDER
    }, 
    'PPIC' : {
        'raw' : Path('calyapo/data/raw/ppic'), 
        'intermediate' : Path('calyapo/data/intermediate/ppic'), 
        'processed' : Path('calyapo/data/processed/ppic'), 
        'penultimate' : UNIVERSAL_PENULTIMATE_FOLDER, 
        'final' : UNIVERSAL_FINAL_FOLDER
    }, 
    'CES' : {
        'raw' : Path('calyapo/data/raw/ces'), 
        'intermediate' : Path('calyapo/data/intermediate/ces'), 
        'processed' : Path('calyapo/data/processed/ces'), 
        'penultimate' : UNIVERSAL_PENULTIMATE_FOLDER, 
        'final' : UNIVERSAL_FINAL_FOLDER
    }
}