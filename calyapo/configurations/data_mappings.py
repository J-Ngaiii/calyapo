# per dataset we have a var2text, opt2text and a map that combines both together working as a pseudo-selector class
IGS_VAR_TO_TEXT = {
    # This dictionary maps column/variable encoding in the raw .csv file to textual description of what the variable means.
    # METHODOLOGY: if I have multiple questions do I validate all at once per individual and just have the llm generate a list of responses, or treat same individual diff question as a whole diff data point? Probably list?
    'Q21' : 'age', 
    'party_reg' : 'partyid', 
    'ID' : 'igs_id',
    # training response:
    'Q27' : 'ideology', 
    # validation response:
    'Q20_1' : 'trump_opinion', 
    # not used yet:
    'Q7_10' : 'abortion_senate', 
    'Q16' : 'prop1_2024', 
}

IGS_OPT_TO_TEXT = {
    # This dictionary maps option encodings in the raw .csv file to textual descriptions of what that option means.
    'age' : {
        '18-29' : [i for i in range(18, 30)], 
        '30-39' : [i for i in range(30, 40)], 
        '40-49' : [i for i in range(40, 50)], 
        '50-64' : [i for i in range(50, 65)], 
        '65+' : [i for i in range(65, 150)]
    }, 
    'partyid' : {
        'Democrat' : 1, 
        'Republican' : 4, 
        'No Party Preference' : 2, 
        'Other Party' : 3
    }, 
    # training response
    'ideology' : {
        'Strongly conservative' : 1, 
        'Somewhat conservative' : 2, 
        'Moderate' : 3, 
        'Somewhat liberal' : 4, 
        'Strongly liberal' : 5
    }, 
    # validation response
    'trump_opinion' : {
        'Strongly favorable' : 1, 
        'Somewhat favorable' : 2, 
        'Somewhat unfavorable' : 3, 
        'Strongly unfavorable' : 4, 
        'No opinion' : 5
    }
}

IGS_MAPS = {
    'maps' : {
        'var2text' : IGS_VAR_TO_TEXT, 
        'opt2text' : IGS_OPT_TO_TEXT
    }, 
}