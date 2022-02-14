# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:09:41 2021

@author: karthiksankar2
"""

import pandas as pd
from joblib import Parallel, delayed
from helper_template_extraction_v1 import get_templates

datasub = pd.read_pickle('Training_Set_Processed_new_topK_v1.pkl')

#Extract all templates:
print("Extracting all templates")
templates=Parallel(n_jobs=20, verbose=1)(delayed(get_templates)(rxn_smi) for rxn_smi in datasub['rxn_smiles'])
datasub["template"]=templates
datasub.to_pickle ('Training_Set_Processed_new_topK_v2.pkl')
