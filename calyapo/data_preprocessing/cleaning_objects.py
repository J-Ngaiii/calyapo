
from pathlib import Path
from typing import List, Dict, Any

class Individual:
    def __init__(self, idx, time_period, dataset_name):
        self.id = idx
        self.time_period = time_period
        self.dataset_name = dataset_name
        self.demog = {}
        self.question_map = {
                            "train" : {
                                "var_label2qst_text": {}, # we keep track of the variable label (`ideology`), question text and question response
                                "var_label2qst_option": {}
                            }, 
                            "val" : {
                                "var_label2qst_text": {},
                                "var_label2qst_option": {}
                            }, 
                            "test" : {
                                "var_label2qst_text": {}, 
                                "var_label2qst_option": {}
                            }, 
                        }
        
    def add_demog(self, var_label, decoded_demog):
        self.demog[var_label] = str(decoded_demog)

    def add_train(self, var_label, question_text, decoded_train):
        self.question_map['train']['var_label2qst_text'][var_label] = str(question_text)
        self.question_map['train']['var_label2qst_option'][var_label] = str(decoded_train)

    def add_val(self, var_label, question_text, decoded_val):
        self.question_map['val']['var_label2qst_text'][var_label] = str(question_text)
        self.question_map['val']['var_label2qst_option'][var_label] = str(decoded_val)
    
    def add_test(self, var_label, question_text, decoded_test):
        self.question_map['test']['var_label2qst_text'][var_label] = str(question_text)
        self.question_map['test']['var_label2qst_option'][var_label] = str(decoded_test)

    def return_full_individual(self):
        entry = {
                "id" : self.id, # default to row index, can use igs id or can just not
                "time" : self.time_period, 
                "demog" : self.demog,
                "dataset" : self.dataset_name,  
                "train" : self.question_map['train'],
                "val" : self.question_map['val'],
                "test" : self.question_map['test'],
            }
        return entry
    
    def return_train_individual(self):
        entry = {
                "id" : self.id, 
                "time" : self.time_period, 
                "demog" : self.demog,
                "dataset" : self.dataset_name,  
                "train" : self.question_map['train']
            }
        return entry
    
    def return_val_individual(self):
        entry = {
                "id" : self.id, 
                "time" : self.time_period, 
                "demog" : self.demog,
                "dataset" : self.dataset_name,  
                "val" : self.question_map['val'],
            }
        return entry
    
    def return_test_individual(self):
        entry = {
                "id" : self.id, 
                "time" : self.time_period, 
                "demog" : self.demog,
                "dataset" : self.dataset_name,  
                "test" : self.question_map['test'],
            }
        return entry
    

class DataPackage:
    def __init__(self, dataset_name, time_period):
        self.dataset_name = dataset_name
        self.time_period = time_period
        self.data_store = {}

    def add_data(self, keyword: str, data: List):
        self.data_store[keyword] = data

    def get_data(self, keyword):
        return self.data_store.get(keyword) 