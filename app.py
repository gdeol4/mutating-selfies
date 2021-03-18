import time
import random
import numpy as np
import streamlit as st

########## RDKIT & SELFIES ##########
import rdkit
from rdkit import Chem
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import AllChem, Mol, Draw
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint, GetBTFingerprint
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

import selfies
from selfies import encoder, decoder

st.set_page_config(
    page_title='Mutating Molecules with a Genetic Algorithm',
    layout='centered',
    page_icon=":octopus:",
    initial_sidebar_state="expanded")

st.sidebar.header("Adjustable Parameters")
st.sidebar.write("A generation is a set of molecules to be evolved")
gen_num = st.sidebar.slider("Generations", min_value=1000, max_value=10000)
st.sidebar.write("Population size of each generation")
gen_size = st.sidebar.slider("Size", min_value=100, max_value=500)
#st.sidebar.write("")
st.sidebar.write('''A run at the lowest setting takes about 30 seconds''')
run_num = st.sidebar.slider("Number of runs", min_value = 1, max_value = 10)



def get_ECFP4(mol):
    ''' Return rdkit ECFP4 fingerprint object for mol

    Parameters:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object

    Returns:
    rdkit ECFP4 fingerprint object for mol
    '''
    return AllChem.GetMorganFingerprint(mol, 2)



def sanitize_smiles(smi):
    '''Return a canonical smile representation of smi

    Parameters:
    smi (string) : smile string to be canonicalized

    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
    smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): True/False to indicate if conversion was  successful
    '''
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)

def mutate_selfie(selfie, max_molecules_len, write_fail_cases=False):
    '''Return a mutated selfie string (only one mutation on slefie is performed)

    Mutations are done until a valid molecule is obtained
    Rules of mutation: With a 50% propbabily, either:
        1. Add a random SELFIE character in the string
        2. Replace a random SELFIE character with another

    Parameters:
    selfie            (string)  : SELFIE string to be mutated
    max_molecules_len (int)     : Mutations of SELFIE string are allowed up to this length
    write_fail_cases  (bool)    : If true, failed mutations are recorded in "selfie_failure_cases.txt"

    Returns:
    selfie_mutated    (string)  : Mutated SELFIE string
    smiles_canon      (string)  : canonical smile of mutated SELFIE string
    '''
    valid=False
    fail_counter = 0
    chars_selfie = get_selfie_chars(selfie)

    while not valid:
        fail_counter += 1

        alphabet = list(selfies.get_semantic_robust_alphabet()) # 34 SELFIE characters

        choice_ls = [1, 2] # 1=Insert; 2=Replace; 3=Delete
        random_choice = np.random.choice(choice_ls, 1)[0]

        # Insert a character in a Random Location
        if random_choice == 1:
            random_index = np.random.randint(len(chars_selfie)+1)
            random_character = np.random.choice(alphabet, size=1)[0]

            selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index:]

        # Replace a random character
        elif random_choice == 2:
            random_index = np.random.randint(len(chars_selfie))
            random_character = np.random.choice(alphabet, size=1)[0]
            if random_index == 0:
                selfie_mutated_chars = [random_character] + chars_selfie[random_index+1:]
            else:
                selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index+1:]

        # Delete a random character
        elif random_choice == 3:
            random_index = np.random.randint(len(chars_selfie))
            if random_index == 0:
                selfie_mutated_chars = chars_selfie[random_index+1:]
            else:
                selfie_mutated_chars = chars_selfie[:random_index] + chars_selfie[random_index+1:]

        else:
            raise Exception('Invalid Operation trying to be performed')

        selfie_mutated = "".join(x for x in selfie_mutated_chars)
        sf = "".join(x for x in chars_selfie)

        try:
            smiles = decoder(selfie_mutated)
            mol, smiles_canon, done = sanitize_smiles(smiles)
            if len(selfie_mutated_chars) > max_molecules_len or smiles_canon=="":
                done = False
            if done:
                valid = True
            else:
                valid = False
        except:
            valid=False
            if fail_counter > 1 and write_fail_cases == True:
                f = open("selfie_failure_cases.txt", "a+")
                f.write('Tried to mutate SELFIE: '+str(sf)+' To Obtain: '+str(selfie_mutated) + '\n')
                f.close()

    return (selfie_mutated, smiles_canon)


def get_selfie_chars(selfie):
    '''Obtain a list of all selfie characters in string selfie

    Parameters:
    selfie (string) : A selfie string - representing a molecule

    Example:
    >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']

    Returns:
    chars_selfie: list of selfie characters present in molecule selfie
    '''
    chars_selfie = [] # A list of all SELFIE sybols from string selfie
    while selfie != '':
        chars_selfie.append(selfie[selfie.find('['): selfie.find(']')+1])
        selfie = selfie[selfie.find(']')+1:]
    return chars_selfie


def get_reward(selfie_A_chars, selfie_B_chars):
    '''Return the levenshtein similarity between the selfies characters in 'selfie_A_chars' & 'selfie_B_chars'


    Parameters:
    selfie_A_chars (list)  : list of characters of a single SELFIES
    selfie_B_chars (list)  : list of characters of a single SELFIES

    Returns:
    reward (int): Levenshtein similarity between the two SELFIES
    '''
    reward = 0
    iter_num = max(len(selfie_A_chars), len(selfie_B_chars)) # Larger of the selfie chars to iterate over

    for i in range(iter_num):

        if i+1 > len(selfie_A_chars) or i+1 > len(selfie_B_chars):
            return reward

        if selfie_A_chars[i] == selfie_B_chars[i]:
            reward += 1

    return reward

# Executable code for EXPERIMENT C (Three different choices):

# TIOTOXENE RUN
# N                  = 20  # Number of runs
# simlr_path_collect = []
# svg_file_name      = 'Tiotixene_run.svg'
# starting_mol_name  = 'Tiotixene'
# data_file_name     = '20_runs_data_Tiotixene.txt'
# starting_smile     = 'CN1CCN(CC1)CCC=C2C3=CC=CC=C3SC4=C2C=C(C=C4)S(=O)(=O)N(C)C'
# show_gen_out       = False
# len_random_struct  = len(get_selfie_chars(encoder(starting_smile))) # Length of the starting SELFIE structure

# CELECOXIB RUN
# N                  = 20  # Number of runs
# simlr_path_collect = []
# svg_file_name      = 'Celecoxib_run.svg'
# starting_mol_name  = 'Celecoxib'
# data_file_name     = '20_runs_data_Celecoxib.txt'
# starting_smile     = 'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'
# show_gen_out       = False
# len_random_struct  = len(get_selfie_chars(encoder(starting_smile))) # Length of the starting SELFIE structure

# Troglitazone RUN
N                  = run_num  # Number of runs
simlr_path_collect = []
svg_file_name      = 'Scopolamine.svg'
starting_mol_name  = 'Scopolamine'
data_file_name     = 'generated molecules.txt'
starting_smile     = 'CC1=C(C2=C(CCC(O2)(C)COC3=CC=C(C=C3)CC4C(=O)NC(=O)S4)C(=C1O)C)C'
show_gen_out       = False
len_random_struct  = len(get_selfie_chars(encoder(starting_smile))) # Length of the starting SELFIE structure


st.title("Mutating drug molecules ")
st.markdown("I made this app using code from the paper *Beyond Generative Models: Superfast Traversal, Optimization, Novelty, Exploration and Discovery (STONED) Algorithm for Molecules using SELFIES*. I used the molecule scopolamine as the starting structure, a phytochemical with anti-parkinsons drug properties. The genetic algorithm is capable of producing novel molecules on par with GPU powered deep learning models without the expensive resources")

#st.sidebar.title("")

if st.sidebar.button("Generate Molecules"):
    for i in range(N):
        print('Run number: ', i)
        with open(data_file_name, 'a') as myfile:
            myfile.write('RUN {} \n'.format(i))

        # celebx = 'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'
        starting_selfie = encoder(starting_smile)
        starting_selfie_chars = get_selfie_chars(starting_selfie)
        max_molecules_len = len(starting_selfie_chars)


        # Random selfie initiation:
        alphabet = list(selfies.get_semantic_robust_alphabet()) # 34 SELFIE characters
        selfie = ''
        for i in range(random.randint(1, len_random_struct)): # max_molecules_len = max random selfie string length
            selfie = selfie + np.random.choice(alphabet, size=1)[0]
        starting_selfie = [selfie]
        print('Starting SELFIE: ', starting_selfie)

        generation_size = gen_size
        num_generations = gen_num
        save_best = []
        simlr_path = []
        reward_path = []

        # Initial set of molecules
        population = np.random.choice(starting_selfie, size=500).tolist() # All molecules are in SELFIES representation


        for gen_ in range(num_generations):

            # Calculate fitness for all of them
            fitness = [get_reward(starting_selfie_chars, get_selfie_chars(x)) for x in population]
            fitness = [float(x)/float(max_molecules_len) for x in fitness] # Between 0 and 1

            # Keep the best member & mutate the rest
            #    Step 1: Keep the best molecule
            best_idx = np.argmax(fitness)
            best_selfie = population[best_idx]

            # Diplay some Outputs:
            if show_gen_out:
                print('Generation: {}/{}'.format(gen_, num_generations))
                print('    Top fitness: ', fitness[best_idx])
                print('    Top SELFIE: ', best_selfie)
            with open(data_file_name, 'a') as myfile:
                myfile.write('    SELFIE: {} FITNESS: {} \n'.format(best_selfie, fitness[best_idx]))


            #    Maybe also print the tanimoto score:
            mol = Chem.MolFromSmiles(decoder(best_selfie))
            target = Chem.MolFromSmiles(starting_smile)
            fp_mol = get_ECFP4(mol)
            fp_target = get_ECFP4(target)
            score = TanimotoSimilarity(fp_mol, fp_target)

            simlr_path.append(score)
            reward_path.append(fitness[best_idx])
            save_best.append(best_selfie)

            #    Step 2: Get mutated selfies
            new_population = []
            for i in range(generation_size-1):
                # selfie_mutated, _ = mutate_selfie(best_selfie, max_molecules_len, write_fail_cases=True)
                selfie_mutated, _ = mutate_selfie(best_selfie, len_random_struct, write_fail_cases=True) # 100 == max_mol_len allowen in mutation
                new_population.append(selfie_mutated)
            new_population.append(best_selfie)

            # Define new population for the next generation
            population = new_population[:]
            if score >= 1:
                print('Limit reached')
                simlr_path_collect.append(simlr_path)
                break

    mols = []
    scores = []
    with open('./generated molecules.txt', 'r') as f:
        for line in f:
            if not 'RUN' in line:
                line = line.rstrip().split()
                slf = line[1]
                score = line[-1]
                m = Chem.MolFromSmiles(selfies.decoder(slf))
                mols.append(m)
                mol = mols[-1:]
                scores.append(score)
                mscore = scores[-1:]

#    st.write(len(mols))
#    st.write(scores[-3:])
#    st.image(Draw.MolsToGridImage(mol, molsPerRow=3, returnPNG=True))
    #st.image(rdkit.Chem.Draw.MolToMPL(mol[0], size=(300, 300), kekulize=True, wedgeBonds=True, imageType=None, fitImage=False, options=None))
    #st.image(rdkit.Chem.Draw.MolsToImage(mols[-1:], subImgSize=(2000, 2000), legends=None))
    #st.image(rdkit.Chem.Draw.MolsToImage(mols[-5:], subImgSize=(4000, 4000), title='RDKit Molecule'))
    #st.image(rdkit.Chem.Draw.MolsToGridImage(mols[-1], molsPerRow=3, subImgSize=(1000, 1000), highlightBondLists=None, useSVG=True))
    scopo = Chem.MolFromSmiles("CC1=C(C2=C(CCC(O2)(C)COC3=CC=C(C=C3)CC4C(=O)NC(=O)S4)C(=C1O)C)C")
    filename = "Scopolamine.png"
    filename2 = "Novel.png"
    novel = mols[-15]

    with st.beta_container():
        st.write("Number of molecules generated: ", len(mols))
        st.write("Similiarty to scopolamine: ", float(scores[-15]))

    with st.beta_container():
        col1,col2 = st.beta_columns(2)

        rdkit.Chem.Draw.MolToImageFile(scopo, filename, size=(2000, 2000), kekulize=True, wedgeBonds=True)
        col1.subheader("Original Molecule")
        col1.image("Scopolamine.png")

        rdkit.Chem.Draw.MolToImageFile(novel, filename2, size=(2000, 2000), kekulize=True, wedgeBonds=True)
        col2.subheader("Mutated Molecule")
        col2.image("Novel.png")
        #col2.image(rdkit.Chem.Draw.MolsToImage(mols[-1:], subImgSize=(2000, 2000), legends=None))
