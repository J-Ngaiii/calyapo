# per dataset we have a var2text, opt2text and a map that combines both together working as a pseudo-selector class
IGS_VAR_TO_LABEL_24 = {
    # This dictionary maps column/variable encoding in the raw .csv file to textual description of what the variable means.
    # METHODOLOGY: if I have multiple questions do I validate all at once per individual and just have the llm generate a list of responses, or treat same individual diff question as a whole diff data point? Probably list?
    
    # we are mapping cols to `variable_labels` these should stay the same across all datasets and time periods
    
    'Q21' : 'age', # NOTE: there's technically multiple age cols but I used this one cuz i could garuantee its respondent age
    'party_reg' : 'partyid', # could also use Q33d
    'ID' : 'dataset_id',
    # training response:
    'Q27' : 'ideology', 
    # validation response:
    'Q20_1' : 'trump_opinion', 
    # not used yet:
    'Q7_10' : 'abortion_senate', 
    'Q16' : 'prop1_2024', 
    'SEX' : 'sex', 
    'Q22' : 'gender', 
    'Q23' : 'education', 
    'Q24_combined' : 'race', 
    'URBANICITY' : 'env_urban'
}

IGS_LABEL_TO_OPT_24 = {
    # This dictionary maps option encodings in the raw .csv file to textual descriptions of what that option means.
    'age' : {
        '18-29' : [str(i) for i in range(18, 30)], 
        '30-39' : [str(i) for i in range(30, 40)], 
        '40-49' : [str(i) for i in range(40, 50)], 
        '50-64' : [str(i) for i in range(50, 65)], 
        '65+' : [str(i) for i in range(65, 150)]
    }, 
    'partyid' : {
        'Democrat' : "1", 
        'Republican' : "4", 
        'No Party Preference' : "2", 
        'Other Party' : "3"
    }, 
    'sex' : {
        'Female' : "1", 
        'Male' : "2"
    }, 
    'gender' : {
        'Male' : "1", 
        'Female' : "2", 
        'Transgender male' : "3", 
        'Transgender female' : "4", 
        'Non-binary' : "5"
    }, 
    'education': {
        '8th grade or less': "1",
        'Some high school': "2",
        'High school graduate': "3",
        'Trade/vocational school': "4",
        '1-2 years college/Associate degree': "5",
        '3 or more years college (no Bachelor\'s degree)': "6",
        'College graduate (Bachelor\'s degree)': "7",
        'Post-graduate degree (e.g., Masters, MD, LLD, PhD, etc.)': "8"
    }, 
    'env_urban': {
        'Rural' : "1", 
        'Suburb/mix' : "2", 
        'Urban' : "3"
    }, 
    # training response
    'ideology' : {
        'Strongly conservative' : "1", 
        'Somewhat conservative' : "2", 
        'Moderate' : "3", 
        'Somewhat liberal' : "4", 
        'Strongly liberal' : "5"
    }, 
    # validation response
    'trump_opinion' : {
        'Strongly favorable' : "1", 
        'Somewhat favorable' : "2", 
        'Somewhat unfavorable' : "3", 
        'Strongly unfavorable' : "4", 
        'No opinion' : "5"
    }, 
    # test response
    'abortion_senate' : {
        'Very important' : "1", 
        'Somewhat important' : "2", 
        'Not a reason or not applicable' : "3", 
        'No opinion' : "4"
    }
}

IGS_LABEL_TO_QES_24 =  {
    'age' : 'What is your age?', 
    'partyid' : 'Party registration from PDI listing',  #NOTE: codebook doesn't actually say the question?
    'ID' : 'igs_id', 
    'ideology' : 'In general, how would you describe your political views?',
    'trump_opinion' : "Regardless of whom you may support in the November presidential election, please indicate whether your opinion of the following candidates is generally favorable or unfavorable, or don't you know enough about him or her to say?  (1)  Donald Trump", 
    'abortion_senate' : 'For each of the following attributes please indicate how important each was to you in your decision to support your preferred candidate in the U.S. Senate election. (10). Will oppose abortion in the Senate', 
    'prop1_2024' : 'If you were voting today, how would you vote on Proposition 1?', 
    'sex' : 'Sex from PDI listing', 
    'gender' : 'How would you describe your gender?', 
    'education' : 'What is the highest year of school that you have finished and gotten credit for?', 
    'race' : 'Which of the following best describes your race?  You may select more than one race, if applicable.', 
    'env_urban' : 'Urbanicity from PDI listing'
}

IGS_MAPS = {
    '2024' : { # mappings may change across years
        'var2label' : IGS_VAR_TO_LABEL_24, 
        'label2opt' : IGS_LABEL_TO_OPT_24, 
        'label2qes' : IGS_LABEL_TO_QES_24
    }

}