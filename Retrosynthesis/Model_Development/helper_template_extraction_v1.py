# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:09:40 2021

@author: karthiksankar2
"""

from template_extractor_enz_v4 import extract_from_reaction

def get_templates(rxn_smi):
    
    # extracts the template    
    try:
        #convert reaction into a dictionary
        reaction = {}
        rct_0, rea_0, prd_0 = rxn_smi.split(' ')[0].split('>')
        reaction['reactants'] = rct_0
        reaction['products'] = prd_0
        reaction['_id'] = 0
                
        #extract the template
        template = extract_from_reaction(reaction)['reaction_smarts']
    
    # fails to extract template
    except:
        template=None
    return template