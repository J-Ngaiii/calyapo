from pathlib import Path

UNIVERSAL_FINAL_FOLDER = Path('calyapo/data/final')
DATA_PATHS = {
    'IGS' : {
        'raw' : Path('calyapo/data/raw/igs'), 
        'processed' : Path('calyapo/data/processed/igs'), 
        'final' : UNIVERSAL_FINAL_FOLDER
    }, 
    'PPIC' : {
        'raw' : Path('calyapo/data/raw/ppic'), 
        'processed' : Path('calyapo/data/processed/ppic'), 
        'final' : UNIVERSAL_FINAL_FOLDER
    }
}