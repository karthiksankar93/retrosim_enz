from template_extractor_enz_v4 import extract_from_reaction
from main_v3 import rdchiralRun
from rdchiral.initialization import rdchiralReaction, rdchiralReactants
from rdkit import DataStructs
import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem

def get_templates(rxn_smi):
    '''Given reaction SMILES, get template'''
    try:
        #convert reaction into a dictionary
        reaction = {}
        rct_0, rea_0, prd_0 = rxn_smi.split(' ')[0].split('>')
        reaction['reactants'] = rct_0
        reaction['products'] = prd_0
        reaction['_id'] = 0
                
        #extract the template
        template = extract_from_reaction(reaction)['reaction_smarts']
    except:
        template=None
    return template

def get_fingerprint(getfp_label, product_smiles):
    '''Given a fingerprint label and product smiles, return its fingerprint'''
    #set fingerprint type
    if getfp_label == 'Morgan2noFeat':
        getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False, useChirality = True)
    elif getfp_label == 'Morgan3noFeat':
        getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 3, useFeatures=False, useChirality = True)
    elif getfp_label == 'Morgan2Feat':
        getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=True, useChirality = True)
    elif getfp_label == 'Morgan3Feat':
        getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 3, useFeatures=True, useChirality = True)
    else:
        raise ValueError('Unknown getfp label')
    
    #given a product smiles, get its fingerprint
    return getfp(product_smiles)

def do_one(similarity_label, getfp_label, datasub, datasub_test_ix, max_prec=40):

    #setup a similarity metric
    if similarity_label == 'Tanimoto':
        similarity_metric = DataStructs.BulkTanimotoSimilarity
    elif similarity_label == 'Dice':
        similarity_metric = DataStructs.BulkDiceSimilarity
    elif similarity_label == 'TverskyA': # weighted towards punishing onlyA
        def similarity_metric(x, y):
            return DataStructs.BulkTverskySimilarity(x, y, 1.5, 1.0)
    elif similarity_label == 'TverskyB': # weighted towards punishing onlyB
        def similarity_metric(x, y):
            return DataStructs.BulkTverskySimilarity(x, y, 1.0, 1.5)
    else:
        raise ValueError('Unknown similarity label')

    #set fingerprint type    
    if getfp_label == 'Morgan2noFeat':
        getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False, useChirality = True)
    elif getfp_label == 'Morgan3noFeat':
        getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 3, useFeatures=False, useChirality = True)
    elif getfp_label == 'Morgan2Feat':
        getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=True, useChirality = True)
    elif getfp_label == 'Morgan3Feat':
        getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 3, useFeatures=True, useChirality = True)
    else:
        raise ValueError('Unknown getfp label')
        
    #loads product SMILES into RDChiral object    
    rct = rdchiralReactants(datasub_test_ix['prod_smiles'])
    
    #get the fingerprint of the product
    fp = datasub_test_ix['prod_fp']
    
    #calculates similarity metric between fingerprint 
    # and all fingerprints in the database
    sims = similarity_metric (fp, [fp_ for fp_ in datasub['prod_fp']])
    
    #sort the similarity metric in reverse order
    js = np.argsort(sims) [::-1]
    
    #get prec_goal from the test dataframe
    prec_goal = datasub_test_ix['prec_goal']
        
    # Get probability of precursors
    probs = {}
    
    for ji,j in enumerate (js[:max_prec]):
        jx = datasub.index[j]
        
        #get template from the training dataset
        template=datasub['template'][jx]
                
        #get rcts reference fingerprint
        rcts_ref_fp = getfp(datasub['rxn_smiles'][jx].split('>')[0])        
        
        #load the template into RDChiralReaction
        try:
            rxn = rdchiralReaction(template)
        except:
            continue 
        
        #get outcomes by running the reaction with the template!
        try:
            outcomes = rdchiralRun(rxn, rct, combine_enantiomers=False)
        
        except:
            outcomes = []
        
        #compute reactant similarity
        for precursors in outcomes:
            precursors_fp = getfp(precursors)
            precursors_sim = similarity_metric(precursors_fp, [rcts_ref_fp])[0]
            if precursors in probs:
                probs[precursors] = max(probs[precursors], precursors_sim * sims[j])
            else:
                probs[precursors] = precursors_sim * sims[j]
    
    #by default, not found
    found_rank = 9999
    
    #check if success criteria is met
    for r, (prec, prob) in enumerate(sorted(probs.items(), key=lambda x:x[1], reverse=True)[:]):      
        #prec_goal is a list of the target reactant, is prec an item in the list?        
        if prec in prec_goal:
            found_rank = min(found_rank,r + 1)
    
    return found_rank
