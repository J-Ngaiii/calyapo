from calyapo.configurations.data_mappings import *
# all mappings
ALL_DATA_MAPS = {
    'IGS' : IGS_MAPS
}

# IGS finetuning variable maps
IGS_IDEO_IDEO = {
    'demo' : ['age', 'partyid'], 
    'train_resp' : ['ideology'], 
    'val_resp' : ['ideology'], 
    'test_resp' : ['ideology']
}

IGS_IDEO_TRUMP =  {
    'demo' : ['age', 'partyid'], 
    'train_resp' : ['ideology'], 
    'val_resp' : ['trump_opinion'], 
    'test_resp' : ['abortion_senate']
}

# all plans
TRAIN_PLANS = {
    # each training plan is defined by specifiying a type of question for training and a type of question for validation
    # eg ideo-ideo means we train on ideology and validate on ideology
    # idea with the TRAIN_PLANS map is that we specifying a training plan then the subdictionary maps the corresponding dataset to its finetuning variable map
    'ideology_to_ideology' : {
        'IGS' : IGS_IDEO_IDEO
    }, 
    'ideology_to_trump' : {
        'IGS' : IGS_IDEO_TRUMP
    }
}