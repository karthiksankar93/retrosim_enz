import pandas as pd
from joblib import Parallel, delayed
#The hard work is done by helper_parallel_v6
from helper_parallel_v6 import do_one, get_fingerprint
import os

SCRIPT_ROOT = os.path.dirname(__file__)
cwd = os.getcwd()

#########DEFINITIONS FOR VALIDATION SEARCH###############
all_getfp_labels = ['Morgan2noFeat', 'Morgan3noFeat','Morgan2Feat','Morgan3Feat']
all_similarity_labels = ['Tanimoto', 'Dice', 'TverskyA', 'TverskyB']

def ranks_to_acc(found_at_rank, fid=None):
    def fprint(txt):
        print(txt)
        if fid is not None:
            fid.write(txt + '\n')
            
    tot = float(len(found_at_rank))
    fprint('{:>8} \t {:>8}'.format('top-n', 'accuracy'))
    accs = []
    for n in [1, 3, 5, 10, 20, 50, 100]:
        accs.append(sum([r <= n for r in found_at_rank]) / tot)
        fprint('{:>8} \t {:>8}'.format(n, accs[-1]))
    return accs

#Load the training dataset
datasub = pd.read_pickle('Training_Set_Processed_new_topK_v2.pkl')

#Load the validation dataset
datasub_test = pd.read_pickle('Validation_Set_Processed_new_topK_v1.pkl')

#Loop through all fingerprint settings
for fp in all_getfp_labels:
    
#loop through all similarity settings
    for sim in all_similarity_labels:

        getfp_label = fp
        similarity_label = sim
        
        print ('Getting fingerprint information for training data')
        fingerprint=Parallel(n_jobs=20, verbose=1)(delayed(get_fingerprint)(getfp_label, product_smiles) for product_smiles in datasub['prod_smiles'])
        datasub['prod_fp'] = fingerprint
        
        print ('Getting fingerprint information for test data')
        fingerprint_test = Parallel(n_jobs=20, verbose=1)(delayed(get_fingerprint)(getfp_label, product_smiles) for product_smiles in datasub_test['prod_smiles'])
        datasub_test['prod_fp'] = fingerprint_test
        
        print("Computing ranks for all test reactions")
        results=Parallel(n_jobs=20, verbose=1)(delayed(do_one) (similarity_label, getfp_label, datasub, datasub_test.iloc[ix]) for ix in datasub_test.index)
        
        #get rank values
        found_at_rank = []
        for ii, ix in enumerate(datasub_test.index):
            found_rank = results[ii]
            found_at_rank.append(found_rank)
        
        #change to out folder
        os.chdir (os.path.join(SCRIPT_ROOT, 'out_val'))
        
        ### Save to individual file
        with open ('{}_{}_{}.txt'.format (getfp_label, similarity_label,'val'), 'w') as fid:
            accs = ranks_to_acc(found_at_rank, fid = fid)
        
        ### Save to global results file
        if not os.path.isfile('results.txt'):
            with open('results.txt', 'w') as fid2:
                fid2.write('\t'.join(['{:>16}'.format(x) for x in [
                'dataset', 'getfp_label', 'similarity_label',
                'top-1 acc', 'top-3 acc', 'top-5 acc', 'top-10 acc',
                'top-20 acc','top-50 acc','top-100 acc']]) + '\n')
        
        with open ('results.txt', 'a') as fid2:
            fid2.write('\t'.join(['{:>16}'.format(x) for x in [
                                    'val', '{}'.format(getfp_label), '{}'.format(similarity_label),
                                ] + accs]) + '\n')
        
        os.chdir (cwd)
