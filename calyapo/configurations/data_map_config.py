from calyapo.configurations.data_mappings import *
# all mappings
ALL_DATA_MAPS = {
    'IGS' : IGS_MAPS
}

# all variable_labels
VARLABEL_DESC = {
    'age': 'age',
    'partyid': 'party identity',
    'dataset_id': 'respondent ID',
    'ideology': 'political ideology',
    'trump_opinion': 'Donald Trump favorability',
    'oppose_abortion_senate': 'importance of Senatorial candidate opposing abortion access',
    'defend_abortion_senate': 'importance of Senatorial candidate defending abortion access',
    'prop1_2024': 'California Proposition 1',
    'sex': 'biological sex',
    'gender': 'gender identity',
    'education': 'highest education level achieved',
    'race': 'racial identity',
    'env_urban': 'degree of residence urbanization', 
    'marital' : 'marital status', 
    'oppose_immigration_senate' : 'importance of Senatorial candidate being tough on immigration',
    'biden_opinion' : 'Joe Biden favorability',
    'harris_opinion' : 'Kamala Harris favorability'   
}

# IGS finetuning variable maps
IDEO_IDEO = {
    'demo' : ['age', 'partyid', 'gender'], 
    'train_resp' : ['ideology'], 
    'val_resp' : ['ideology'], 
    'test_resp' : ['ideology']
}

IDEO_TRUMP =  {
    'demo' : ['age', 'partyid', 'gender', 'sex', 'education',  'env_urban', 'race', 'marital'], 
    'train_resp' : ['ideology'], 
    'val_resp' : ['trump_opinion'], 
    'test_resp' : ['trump_opinion']
}

PREZ_ABORTION =  {
    'demo' : ['age', 'partyid', 'ideology', 'race', 'gender', 'sex', 'env_urban', 'marital'], 
    'train_resp' : ['trump_opinion', 'biden_opinion'], 
    'val_resp' : ['oppose_abortion_senate', 'defend_abortion_senate'], 
    'test_resp' : ['oppose_abortion_senate', 'defend_abortion_senate']
} # simulate low information voter that doesn't have policy stances

OPINION_SCHOOL = {
    'demo' : ['age', 'partyid', 'ideology', 'race', 'gender', 'sex', 'env_urban', 'marital'], 
    'train_resp' : ['trump_opinion', 'biden_opinion', 'harris_opinion'], 
    'val_resp' : ['trump_opinion', 'biden_opinion', 'harris_opinion'], 
    'test_resp' : ['trump_opinion', 'biden_opinion', 'harris_opinion']
}

TEST_PLAN = {
    'demo' : ['age', 'partyid', 'ideology', 'race', 'gender', 'sex', 'env_urban', 'marital'], 
    'train_resp' : ['trump_opinion', 'biden_opinion'], 
    'val_resp' : ['oppose_immigration_senate'], 
    'test_resp' : ['oppose_immigration_senate']
}

TEST_PLAN_CES = {
    'demo' : ['age', 'partyid', 'ideology', 'race', 'gender', 'sex', 'env_urban', 'marital'], 
    'train_resp' : [], 
    'val_resp' : [], 
    'test_resp' : []
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
            'question_varies_by_split' : False, # we can do ideo to ideo but have the question wording change
            'train_setting' : 1
        },
    'ideology_to_trump' : {
            'variable_map': IDEO_TRUMP, 
            'homogenous_var_plan' : False, 
            'datasets' : set(['IGS']), 
            'question_varies_by_split' : True, # of course will be true
            'train_setting' : 2
        }, 
    'presidents_to_abortion' : {
            'variable_map': PREZ_ABORTION, 
            'homogenous_var_plan' : False, 
            'datasets' : set(['IGS']), 
            'question_varies_by_split' : True, # of course will be true
            'train_setting' : 2, 
            'valid_indiv_setting' : 'any'
        }, 
    'test_plan' : {
            'variable_map': TEST_PLAN, 
            'homogenous_var_plan' : False, 
            'datasets' : set(['IGS']), 
            'question_varies_by_split' : True, # of course will be true
            'train_setting' : 2, 
            'valid_indiv_setting' : 'any', 
            'reduction_modifier' : 0.1
        }, 
    'test_plan_ces' : {
            'variable_map': TEST_PLAN_CES, 
            'homogenous_var_plan' : True, 
            'datasets' : set(['CES']), 
            'question_varies_by_split' : False, # of course will be true
            'train_setting' : 1, 
            'valid_indiv_setting' : 'any', 
        }, 
    'opinion_school' : {
            'variable_map': OPINION_SCHOOL, 
            'homogenous_var_plan' : True, 
            'datasets' : set(['IGS']), 
            'question_varies_by_split' : False,
            'train_setting' : 1, 
            'valid_indiv_setting' : 'any'
        }, 
}