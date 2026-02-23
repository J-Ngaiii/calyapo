from calyapo.configurations.data_mappings import *
# all mappings
ALL_DATA_MAPS = {
    'IGS' : IGS_MAPS
}

# all variable_labels
VARLABEL_DESC = {
    'age': 'Age',
    'partyid': 'Party Identity',
    'dataset_id': 'Respondent ID',
    'ideology': 'Political Ideology',
    'trump_opinion': 'Trump Favorability',
    'abortion_senate': 'Politician Abortion Stance',
    'prop1_2024': 'California Proposition 1',
    'sex': 'Biological Sex',
    'gender': 'Gender Identity',
    'education': 'Highest Education Level',
    'race': 'Race',
    'env_urban': 'Residence Urbanicity'
}

# IGS finetuning variable maps
IDEO_IDEO = {
    'demo' : ['age', 'partyid', 'gender'], 
    'train_resp' : ['ideology'], 
    'val_resp' : ['ideology'], 
    'test_resp' : ['ideology']
}

IDEO_TRUMP =  {
    'demo' : ['age', 'partyid', 'gender', 'sex', 'env_urban'], 
    'train_resp' : ['ideology'], 
    'val_resp' : ['trump_opinion'], 
    'test_resp' : ['abortion_senate']
}

# all plans
TRAIN_PLANS = {
    # each training plan is defined by specifiying a type of question for training and a type of question for validation
    # eg ideo-ideo means we train on ideology and validate on ideology
    # idea with the TRAIN_PLANS map is that we specifying a training plan then the subdictionary maps the corresponding dataset to its finetuning variable map
    'ideology_to_ideology' : 
        {
            'variable_map': IDEO_IDEO, 
            'homogenous_var_plan' : True, 
            'datasets' : set(['IGS']), 
            'question_varies_by_split' : False # we can do ideo to ideo but have the question wording change
        },
    'ideology_to_trump' : {
            'variable_map': IDEO_TRUMP, 
            'homogenous_var_plan' : False, 
            'datasets' : set(['IGS']), 
            'question_varies_by_split' : True # of course will be true
        }
}