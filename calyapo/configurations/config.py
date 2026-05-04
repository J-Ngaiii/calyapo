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
        'final' : UNIVERSAL_FINAL_FOLDER, 
        'filetypes' : ['csv', 'dta']
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

IGS_RACE_MAP = {
    '202402' : {
        'Q24_1': '1',
        'Q24_2': '2',
        'Q24_3': '3',
        'Q24_4': '4',
        'Q24_5': '5',
        'Q24_6': '6',
        'Q24_7': '7'
    }, 
    '20240529' : {
        'Q43_1': '1',
        'Q43_2': '2',
        'Q43_3': '3',
        'Q43_4': '4',
        'Q43_5': '5',
        'Q43_6': '6',
        'Q43_7': '7', 
        'Q43_8': '8'
    }, 
    '20240819' : {
        'Q38_1': '1',
        'Q38_2': '2',
        'Q38_3': '3',
        'Q38_4': '4',
        'Q38_5': '5',
        'Q38_6': '6',
        'Q38_7': '7', 
        'Q38_8': '8'
    }, 
    '20240925' : {
        'Q34_1': '1',
        'Q34_2': '2',
        'Q34_3': '3',
        'Q34_4': '4',
        'Q34_5': '5',
        'Q34_6': '6',
        'Q34_7': '7', 
        'Q34_8': '8'
    }, 
    '20241028' : {
        'Q24_1': '1',
        'Q24_2': '2',
        'Q24_3': '3',
        'Q24_4': '4',
        'Q24_5': '5',
        'Q24_6': '6',
        'Q24_7': '7', 
        'Q24_8': '8'
    }
}

IGS_SURVEY_WAVE_DESC = {
    '202402' : "Febuary 2024", 
    '20240819' : "August, 2024", 
    '20240925' : "September, 2024", 
    '20241028' : "October, 2024", 
}

POLLING_FIRM_DESC = {
    'IGS' : 'Berkeley Institute of Governmental Studies'
}