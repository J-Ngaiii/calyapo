# per dataset we have a var2text, opt2text and a map that combines both together working as a pseudo-selector class
IGS_VAR_TO_LABEL_2402 = {
    # This dictionary maps column/variable encoding in the raw .csv file to textual description of what the variable means.
    # METHODOLOGY: if I have multiple questions do I validate all at once per individual and just have the llm generate a list of responses, or treat same individual diff question as a whole diff data point? Probably list?
    
    # we are mapping cols to `variable_labels` these should stay the same across all datasets and time periods
    
    'Q21' : 'age', # NOTE: there's technically multiple age cols but I used this one cuz i could garuantee its respondent age
    'party_reg' : 'partyid', # could also use Q33d
    'ID' : 'dataset_id',
    'Q27' : 'ideology', 
    'Q20_1' : 'trump_opinion', 
    'Q7_10' : 'oppose_abortion_senate', 
    'Q15_5' : 'defend_abortion_senate', 
    'Q16' : 'prop1_2024', 
    'SEX' : 'sex', 
    'Q22' : 'gender', 
    'Q23' : 'education', 
    'racial_id' : 'race', 
    'URBANICITY' : 'env_urban', 
    'Q34' : 'marital', 
    'Q7_9' : 'oppose_immigration_senate', 
    'Q20_2' : 'biden_opinion',
}

IGS_LABEL_TO_OPT_2402 = {
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
    'race' : {
        'White': "1", 
        'Black/African American' : "2",
        'Hispanic/Latino' : "3",
        'Asian/Asian American' : "4",
        'Native American/Alaska Native' : "5",
        'Native Hawaiian/Pacific Islander' : "6",
        'Other' : "7"
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
    'marital' : {
        'Married' : "1",
        'Not married, but live together' : "2",
        'Separated or divorced' : "3",
        'Widowed' : "4",
        'Single, never married' : "5"
    },
    'ideology' : {
        'Strongly conservative' : "1", 
        'Somewhat conservative' : "2", 
        'Moderate' : "3", 
        'Somewhat liberal' : "4", 
        'Strongly liberal' : "5"
    }, 
    'trump_opinion' : {
        'Strongly favorable' : "1", 
        'Somewhat favorable' : "2", 
        'Somewhat unfavorable' : "3", 
        'Strongly unfavorable' : "4", 
        'No opinion' : "5"
    }, 
    'oppose_abortion_senate' : {
        'Very important' : "1", 
        'Somewhat important' : "2", 
        'Not a reason or not applicable' : "3", 
        'No opinion' : "4"
    }, 
    'defend_abortion_senate' : {
        'Very important' : "1", 
        'Somewhat important' : "2", 
        'Not a reason or not applicable' : "3", 
        'No opinion' : "4"
    }, 
    'oppose_immigration_senate' : {
        'Very important' : "1", 
        'Somewhat important' : "2", 
        'Not a reason or not applicable' : "3", 
        'No opinion' : "4"
    },
    'biden_opinion' : {
        'Strongly favorable' : "1", 
        'Somewhat favorable' : "2", 
        'Somewhat unfavorable' : "3", 
        'Strongly unfavorable' : "4", 
        'No opinion' : "5"
    },
}

IGS_LABEL_TO_QES_2402 =  {
    'age' : 'What is your age?', 
    'partyid' : 'Party registration from PDI listing',  #NOTE: codebook doesn't actually say the question?
    'ID' : 'igs_id', 
    'ideology' : 'In general, how would you describe your political views?',
    'trump_opinion' : "Regardless of whom you may support in the November presidential election, please indicate whether your opinion of the following candidates is generally favorable or unfavorable, or don't you know enough about him or her to say?  (1)  Donald Trump", 
    'oppose_abortion_senate' : 'For each of the following attributes please indicate how important each was to you in your decision to support your preferred candidate in the U.S. Senate election. (10). Will oppose abortion in the Senate', 
    'defend_abortion_senate' : 'For each of the following attributes please indicate how important each was to you in your decision to support your preferred candidate in the U.S. Senate election. (5).  Would be a strong voice in defending abortion rights for women in the Senate', 
    'prop1_2024' : 'If you were voting today, how would you vote on Proposition 1?', 
    'sex' : 'Sex from PDI listing', 
    'gender' : 'How would you describe your gender?', 
    'education' : 'What is the highest year of school that you have finished and gotten credit for?', 
    'race' : 'Which of the following best describes your race?  You may select more than one race, if applicable.', 
    'env_urban' : 'Urbanicity from PDI listing', 
    'marital' : 'Which of the following best describes your present marital status?',
    'oppose_immigration_senate' : 'For each of the following attributes please indicate how important each was to you in your decision to support your preferred candidate in the U.S. Senate election. (9). Supports tougher immigration laws',
    'biden_opinion' : "Regardless of whom you may support in the November presidential election, please indicate whether your opinion of the following candidates is generally favorable or unfavorable, or don't you know enough about him or her to say? (2) Joe Biden",
}

# ---------------
# 29/05/2024 IGS
# NOT ENOUGH RELEVANT QUESTIONS
# ---------------

IGS_VAR_TO_LABEL_240529 = {
    'Q40' : 'age',
    'pid3' : 'partyid', 
    'ID' : 'dataset_id',
}

IGS_LABEL_TO_OPT_240529 = {
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
}

IGS_LABEL_TO_QES_240529 =  {
    'age' : 'What is your age?', 
    'partyid' : 'Party registration from PDI listing',  
    'ID' : 'igs_id', 
}

# ---------------
# 19/08/2024 IGS
# ---------------

IGS_VAR_TO_LABEL_240819 = {
    'ID' : 'dataset_id',
    'Q35' : 'age', 
    'party_reg' : 'partyid', 
    'Q47' : 'ideology', 
    'racial_id' : 'race', 
    'Q36' : 'gender', 
    'SEX' : 'sex', 
    'URBANICITY' : 'env_urban', 
    'Q50' : 'marital',
    'Q37' : 'education',  
    'Q2_1' : 'trump_opinion', 
    'Q2_2' : 'biden_opinion', 
    'Q16_4' : 'defend_abortion_senate', 
    'Q16_6' : 'oppose_immigration_senate'
}

IGS_LABEL_TO_OPT_240819 = {
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
    'ideology' : {
        'Strongly conservative' : "1", 
        'Somewhat conservative' : "2", 
        'Moderate' : "3", 
        'Somewhat liberal' : "4", 
        'Strongly liberal' : "5"
    }, 
    'race' : {
        'White': "1", 
        'Hispanic or Latino' : "2",
        'Asian' : "3",
        'Native Hawaiian or Pacific Islander' : "4",
        'Black or African American' : "5",
        'Middle Eastern or North African' : "6",
        'American Indian or Alaska Native' : "7", 
        'Other' : "8"
    }, 
    'gender' : {
        'Male' : "1", 
        'Female' : "2", 
        'Transgender male' : "3", 
        'Transgender female' : "4", 
        'Non-binary' : "5"
    }, 
    'sex' : {
        'Female' : "1", 
        'Male' : "2"
    }, 
    'env_urban': {
        'Rural' : "1", 
        'Suburb/mix' : "2", 
        'Urban' : "3"
    }, 
    'marital' : {
        'Married' : "1",
        'Not married, but live together' : "2",
        'Separated or divorced' : "3",
        'Widowed' : "4",
        'Single, never married' : "5"
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
    'trump_opinion' : {
        'Strongly favorable' : "1", 
        'Somewhat favorable' : "2", 
        'Somewhat unfavorable' : "3", 
        'Strongly unfavorable' : "4", 
        'No opinion' : "5"
    }, 
    'biden_opinion' : {
        'Strongly favorable' : "1", 
        'Somewhat favorable' : "2", 
        'Somewhat unfavorable' : "3", 
        'Strongly unfavorable' : "4", 
        'No opinion' : "5"
    },
    'defend_abortion_senate' : {
        'Very important' : "1", 
        'Somewhat important' : "2", 
        'Not a reason or not applicable' : "3", 
        'No opinion' : "4"
    }, 
    'oppose_immigration_senate' : {
        'Very important' : "1", 
        'Somewhat important' : "2", 
        'Not a reason or not applicable' : "3", 
        'No opinion' : "4"
    },
}

IGS_LABEL_TO_QES_240819 =  {
    'age' : 'What is your age?', 
    'partyid' : 'Party registration from PDI listing',  
    'ID' : 'igs_id', 
    'race' : 'Which of the following best describes your race?  You may select more than one race, if applicable.', 
    'gender' : 'How would you describe your gender?', 
    'sex' : 'Sex from PDI listing', 
    'env_urban' : 'Urbanicity from PDI listing', 
    'marital' : 'Which of the following best describes your present marital status?',
    'education' : 'What is the highest year of school that you have finished and gotten credit for?', 
    'trump_opinion' : "Please indicate whether your opinion of the following political figures is favorable or unfavorable, or whether you don't yet know enough about them to offer an opinion?  (1) Donald Trump, former President", 
    'biden_opinion' : "Please indicate whether your opinion of the following political figures is favorable or unfavorable, or whether you don't yet know enough about them to offer an opinion?  (2) Joe Biden, President", 
    'defend_abortion_senate' : 'For each of the following attributes please indicate how important each will be to you when considering whom to support in the U.S. Senate election.   (4).  Would be a strong voice in defending abortion rights for women in the Senate', 
    'oppose_immigration_senate' : 'For each of the following attributes please indicate how important each will be to you when considering whom to support in the U.S. Senate election.   (6).  Supports tougher immigration laws'
}

# ---------------
# 25/09/2024 IGS
# ---------------

IGS_VAR_TO_LABEL_240925 = {
    'ID' : 'dataset_id',
    'Q31' : 'age', 
    'party_reg' : 'partyid', 
    'Q41' : 'ideology', 
    'racial_id' : 'race', 
    'Q32' : 'gender', 
    'SEX' : 'sex', 
    'URBANICITY' : 'env_urban', 
    'Q44' : 'marital', 
    'Q33' : 'education',
    'Q2' : 'prez_vote', 
    'Q3' : 'prez_enthusiasm', 
    'Q4_1' : 'trump_opinion', 
    'Q4_2' : 'harris_opinion', 
}

IGS_LABEL_TO_OPT_240925 = {
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
    'ideology' : {
        'Strongly conservative' : "1", 
        'Somewhat conservative' : "2", 
        'Moderate' : "3", 
        'Somewhat liberal' : "4", 
        'Strongly liberal' : "5"
    }, 
    'race' : {
        'White': "1", 
        'Hispanic or Latino' : "2",
        'Asian' : "3",
        'Native Hawaiian or Pacific Islander' : "4",
        'Black or African American' : "5",
        'Middle Eastern or North African' : "6",
        'American Indian or Alaska Native' : "7", 
        'Other' : "8"
    }, 
    'gender' : {
        'Male' : "1", 
        'Female' : "2", 
        'Transgender male' : "3", 
        'Transgender female' : "4", 
        'Non-binary' : "5"
    }, 
    'sex' : {
        'Female' : "1", 
        'Male' : "2"
    }, 
    'env_urban': {
        'Rural' : "1", 
        'Suburb/mix' : "2", 
        'Urban' : "3"
    }, 
    'marital' : {
        'Married' : "1",
        'Not married, but live together' : "2",
        'Separated or divorced' : "3",
        'Widowed' : "4",
        'Single, never married' : "5"
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
    'prez_vote' : {
        'Kamala Harris, Democrat' : "1", 
        'Donald J. Trump, Republican' : "2", 
        'Robert F. Kennedy, Jr., independent' : "3", 
        'Cornel West, independent' : "4", 
        'Jill Stein, Green Party' : "5", 
        'Chase Oliver, Libertarian Party' : "6", 
        'Undecided' : "7"
    },
    'prez_enthusiasm' : {
        'Extremely enthusiastic' : "1", 
        'Very enthusiastic' : "2", 
        'Somewhat enthusiastic' : "3", 
        'Not too enthusiastic' : "4", 
        'Not at all enthusiastic' : "5", 
        'No difference' : "6"
    }, 
    'trump_opinion' : {
        'Strongly favorable' : "1", 
        'Somewhat favorable' : "2", 
        'Somewhat unfavorable' : "3", 
        'Strongly unfavorable' : "4", 
        'No opinion' : "5"
    },
    'harris_opinion' : {
        'Strongly favorable' : "1", 
        'Somewhat favorable' : "2", 
        'Somewhat unfavorable' : "3", 
        'Strongly unfavorable' : "4", 
        'No opinion' : "5"
    }
}

IGS_LABEL_TO_QES_240925 =  {
    'age' : 'What is your age?', 
    'partyid' : 'Party registration from PDI listing',  
    'ID' : 'igs_id', 
    'race' : 'Which of the following best describes your race?  You may select more than one race, if applicable.', 
    'gender' : 'How would you describe your gender?', 
    'sex' : 'Sex from PDI listing', 
    'env_urban' : 'Urbanicity from PDI listing', 
    'marital' : 'Which of the following best describes your present marital status?',
    'education' : 'What is the highest year of school that you have finished and gotten credit for?', 
    'prez_vote' : 'In the November general election for President, the names of the following candidates will appear on the California election ballot. If the election were held today, for whom would you vote?',
    'prez_enthusiasm' : 'How enthusiastic are you about the presidential candidate that you are supporting?', 
    'trump_opinion' : "Regardless of whom you are supporting, please indicate whether your opinion of the following candidate is favorable or unfavorable, or whether you don't yet know enough about them to offer an opinion? (1) Donald Trump, former President",
    'harris_opinion' : "Regardless of whom you are supporting, please indicate whether your opinion of the following candidate is favorable or unfavorable, or whether you don't yet know enough about them to offer an opinion? (2) Kamala Harris, Vice President"
}

# ---------------
# 28/10/2024 IGS
# ---------------

IGS_VAR_TO_LABEL_241028 = {
    'ID' : 'dataset_id',
    'Q21' : 'age', 
    'party_reg' : 'partyid', 
    'Q31' : 'ideology', 
    'racial_id' : 'race', 
    'Q22' : 'gender', 
    'SEX' : 'sex', 
    'URBANICITY' : 'env_urban', 
    'Q34' : 'marital', 
    'Q23' : 'education',
    'Q4' : 'prez_vote', 
    'Q5_1' : 'trump_opinion', 
    'Q5_2' : 'harris_opinion', 
}

IGS_LABEL_TO_OPT_241028 = {
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
    'ideology' : {
        'Strongly conservative' : "1", 
        'Somewhat conservative' : "2", 
        'Moderate' : "3", 
        'Somewhat liberal' : "4", 
        'Strongly liberal' : "5"
    }, 
    'race' : {
        'White': "1", 
        'Hispanic or Latino' : "2",
        'Asian' : "3",
        'Native Hawaiian or Pacific Islander' : "4",
        'Black or African American' : "5",
        'Middle Eastern or North African' : "6",
        'American Indian or Alaska Native' : "7", 
        'Other' : "8"
    }, 
    'gender' : {
        'Male' : "1", 
        'Female' : "2", 
        'Transgender male' : "3", 
        'Transgender female' : "4", 
        'Non-binary' : "5"
    }, 
    'sex' : {
        'Female' : "1", 
        'Male' : "2"
    }, 
    'env_urban': {
        'Rural' : "1", 
        'Suburb/mix' : "2", 
        'Urban' : "3"
    }, 
    'marital' : {
        'Married' : "1",
        'Not married, but live together' : "2",
        'Separated or divorced' : "3",
        'Widowed' : "4",
        'Single, never married' : "5"
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
    'trump_opinion' : {
        'Strongly favorable' : "1", 
        'Somewhat favorable' : "2", 
        'Somewhat unfavorable' : "3", 
        'Strongly unfavorable' : "4", 
        'No opinion' : "5"
    },
    'harris_opinion' : {
        'Strongly favorable' : "1", 
        'Somewhat favorable' : "2", 
        'Somewhat unfavorable' : "3", 
        'Strongly unfavorable' : "4", 
        'No opinion' : "5"
    }
}

IGS_LABEL_TO_QES_241028 =  {
    'age' : 'What is your age?', 
    'partyid' : 'Party registration from PDI listing',  
    'ID' : 'igs_id', 
    'race' : 'Which of the following best describes your race?  You may select more than one race, if applicable.', 
    'gender' : 'How would you describe your gender?', 
    'sex' : 'Sex from PDI listing', 
    'env_urban' : 'Urbanicity from PDI listing', 
    'marital' : 'Which of the following best describes your present marital status?',
    'education' : 'What is the highest year of school that you have finished and gotten credit for?', 
    'trump_opinion' : "Regardless of whom you are supporting, please indicate whether your opinion of the following candidate is favorable or unfavorable, or whether you don't yet know enough about them to offer an opinion? (1) Donald Trump, former President",
    'harris_opinion' : "Regardless of whom you are supporting, please indicate whether your opinion of the following candidate is favorable or unfavorable, or whether you don't yet know enough about them to offer an opinion? (2) Kamala Harris, Vice President"
}

IGS_MAPS = {
    '202402' : { # mappings may change across years
        'var2label' : IGS_VAR_TO_LABEL_2402, 
        'label2opt' : IGS_LABEL_TO_OPT_2402, 
        'label2qes' : IGS_LABEL_TO_QES_2402
    }, 
    # '20240529' : { 
    #     'var2label' : IGS_VAR_TO_LABEL_240529, 
    #     'label2opt' : IGS_LABEL_TO_OPT_240529, 
    #     'label2qes' : IGS_LABEL_TO_QES_240529
    # }, 
    '20240819' : { 
        'var2label' : IGS_VAR_TO_LABEL_240819, 
        'label2opt' : IGS_LABEL_TO_OPT_240819, 
        'label2qes' : IGS_LABEL_TO_QES_240819
    }, 
    '20240925' : { 
        'var2label' : IGS_VAR_TO_LABEL_240925, 
        'label2opt' : IGS_LABEL_TO_OPT_240925, 
        'label2qes' : IGS_LABEL_TO_QES_240925
    }, 
    '20241028' : { 
        'var2label' : IGS_VAR_TO_LABEL_241028, 
        'label2opt' : IGS_LABEL_TO_OPT_241028, 
        'label2qes' : IGS_LABEL_TO_QES_241028
    }, 
}

CES_VAR_TO_LABEL_24 = {
    'caseid' : 'dataset_id',
    'calculated_age' : 'age', 
    'party_reg' : 'partyid', 
    'Q47' : 'ideology', 
    'racial_id' : 'race', 
    'Q36' : 'gender', 
    'SEX' : 'sex', 
    'URBANICITY' : 'env_urban', 
    'Q50' : 'marital',
    'Q37' : 'education',  
    'Q2_1' : 'trump_opinion', 
    'Q2_2' : 'biden_opinion', 
    'Q16_4' : 'defend_abortion_senate', 
    'Q16_6' : 'oppose_immigration_senate'
}

CES_LABEL_TO_OPT_24 = {
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
    'ideology' : {
        'Strongly conservative' : "1", 
        'Somewhat conservative' : "2", 
        'Moderate' : "3", 
        'Somewhat liberal' : "4", 
        'Strongly liberal' : "5"
    }, 
    'race' : {
        'White': "1", 
        'Hispanic or Latino' : "2",
        'Asian' : "3",
        'Native Hawaiian or Pacific Islander' : "4",
        'Black or African American' : "5",
        'Middle Eastern or North African' : "6",
        'American Indian or Alaska Native' : "7", 
        'Other' : "8"
    }, 
    'gender' : {
        'Male' : "1", 
        'Female' : "2", 
        'Transgender male' : "3", 
        'Transgender female' : "4", 
        'Non-binary' : "5"
    }, 
    'sex' : {
        'Female' : "1", 
        'Male' : "2"
    }, 
    'env_urban': {
        'Rural' : "1", 
        'Suburb/mix' : "2", 
        'Urban' : "3"
    }, 
    'marital' : {
        'Married' : "1",
        'Not married, but live together' : "2",
        'Separated or divorced' : "3",
        'Widowed' : "4",
        'Single, never married' : "5"
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
    'trump_opinion' : {
        'Strongly favorable' : "1", 
        'Somewhat favorable' : "2", 
        'Somewhat unfavorable' : "3", 
        'Strongly unfavorable' : "4", 
        'No opinion' : "5"
    }, 
    'biden_opinion' : {
        'Strongly favorable' : "1", 
        'Somewhat favorable' : "2", 
        'Somewhat unfavorable' : "3", 
        'Strongly unfavorable' : "4", 
        'No opinion' : "5"
    },
    'defend_abortion_senate' : {
        'Very important' : "1", 
        'Somewhat important' : "2", 
        'Not a reason or not applicable' : "3", 
        'No opinion' : "4"
    }, 
    'oppose_immigration_senate' : {
        'Very important' : "1", 
        'Somewhat important' : "2", 
        'Not a reason or not applicable' : "3", 
        'No opinion' : "4"
    },
}

CES_LABEL_TO_QES_24 =  {
    'age' : 'What is your age?', 
    'partyid' : 'Party registration from PDI listing',  
    'ID' : 'igs_id', 
    'race' : 'Which of the following best describes your race?  You may select more than one race, if applicable.', 
    'gender' : 'How would you describe your gender?', 
    'sex' : 'Sex from PDI listing', 
    'env_urban' : 'Urbanicity from PDI listing', 
    'marital' : 'Which of the following best describes your present marital status?',
    'education' : 'What is the highest year of school that you have finished and gotten credit for?', 
    'trump_opinion' : "Please indicate whether your opinion of the following political figures is favorable or unfavorable, or whether you don't yet know enough about them to offer an opinion?  (1) Donald Trump, former President", 
    'biden_opinion' : "Please indicate whether your opinion of the following political figures is favorable or unfavorable, or whether you don't yet know enough about them to offer an opinion?  (2) Joe Biden, President", 
    'defend_abortion_senate' : 'For each of the following attributes please indicate how important each will be to you when considering whom to support in the U.S. Senate election.   (4).  Would be a strong voice in defending abortion rights for women in the Senate', 
    'oppose_immigration_senate' : 'For each of the following attributes please indicate how important each will be to you when considering whom to support in the U.S. Senate election.   (6).  Supports tougher immigration laws'
}

CES_MAPS = {
    '202402' : { # mappings may change across years
        'var2label' : IGS_VAR_TO_LABEL_2402, 
        'label2opt' : IGS_LABEL_TO_OPT_2402, 
        'label2qes' : IGS_LABEL_TO_QES_2402
    }, 
    # '20240529' : { 
    #     'var2label' : IGS_VAR_TO_LABEL_240529, 
    #     'label2opt' : IGS_LABEL_TO_OPT_240529, 
    #     'label2qes' : IGS_LABEL_TO_QES_240529
    # }, 
    '20240819' : { 
        'var2label' : IGS_VAR_TO_LABEL_240819, 
        'label2opt' : IGS_LABEL_TO_OPT_240819, 
        'label2qes' : IGS_LABEL_TO_QES_240819
    }, 
    '20240925' : { 
        'var2label' : IGS_VAR_TO_LABEL_240925, 
        'label2opt' : IGS_LABEL_TO_OPT_240925, 
        'label2qes' : IGS_LABEL_TO_QES_240925
    }, 
    '20241028' : { 
        'var2label' : IGS_VAR_TO_LABEL_241028, 
        'label2opt' : IGS_LABEL_TO_OPT_241028, 
        'label2qes' : IGS_LABEL_TO_QES_241028
    }, 
}