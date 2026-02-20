from pathlib import Path

UNIVERSAL_FINAL_FOLDER = Path('calyapo/data/final')
UNIVERSAL_NA_FILLER = "not available"
DATA_PATHS = {
    'IGS' : {
        'raw' : Path('calyapo/data/raw/igs'), 
        'intermediate' : Path('calyapo/data/intermediate/igs'), 
        'processed' : Path('calyapo/data/processed/igs'), 
        'final' : UNIVERSAL_FINAL_FOLDER
    }, 
    'PPIC' : {
        'raw' : Path('calyapo/data/raw/ppic'), 
        'intermediate' : Path('calyapo/data/intermediate/ppic'), 
        'processed' : Path('calyapo/data/processed/ppic'), 
        'final' : UNIVERSAL_FINAL_FOLDER
    }, 
    'CES' : {
        'raw' : Path('calyapo/data/raw/ces'), 
        'intermediate' : Path('calyapo/data/intermediate/ces'), 
        'processed' : Path('calyapo/data/processed/ces'), 
        'final' : UNIVERSAL_FINAL_FOLDER
    }
}