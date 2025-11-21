IGS_DEMO_VAR_TO_TEXT = {
    # This dictionary maps column/variable encoding in the raw .csv file to textual description of what the variable means.
    'Q21' : 'age', 
    'party_reg' : 'partyid', 
    'ID' : 'igs_id',
}

IGS_RESP_VAR_TO_TEXT = {
    'Q27' : 'ideology'
}

IGS_DEMO_OPT_TO_TEXT = {
    # This dictionary maps option encodings in the raw .csv file to textual descriptions of what that option means. 
    'age' : {
        '18-29' : [i for i in range(18, 30)]
    }, 
    'partyid' : {
        'Democrat' : 1, 
        'Republican' : 4, 
        'No Party Preference' : 2, 
        'Other Party' : 3
    }
}

IGS_RESP_OPT_TO_TEXT = {

}

IGS_MAPS = {
    # all the relevant mappings for IGS
    'demo' : {
        'var2text' : IGS_DEMO_VAR_TO_TEXT, 
        'opt2text' : IGS_DEMO_OPT_TO_TEXT
    }, 
    'resp' : {
        'var2text' : IGS_RESP_VAR_TO_TEXT, 
        'opt2text' : IGS_RESP_OPT_TO_TEXT
    }
}

FULL_DATA_MAPS = {
    'IGS' : IGS_MAPS
}