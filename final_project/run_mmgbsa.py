#!/usr/bin/env python3

#########################
import numpy as np 
import mdtraj as md
import os, sys 
import subprocess as sp
import argparse
##########################

# Dependencies: 
# - amber!

# Tommy's <3
import warnings
import functools

def num_str(s, return_str=True, return_num=True):
    
    s = ''.join(filter(str.isdigit, s))

    if return_str and return_num:
        return s, int(s)

    if return_str:
        return s

    if return_num:
        return int(s)

def sort_strs(strs: list, max=False, indexed: bool=False):
    
    """ strs ::: a list or numpy array of strings.
        max ::: bool, to sort in terms of greatest index to smallest.
        indexed ::: bool, whether or not to filter out strings that don't contain digits.
                    if set to False and string list (strs) contains strings without a digit, function 
                    will return unsorted string list (strs) as an alternative to throwing an error."""
    
    # we have to ensure that each str in strs contains a number otherwise we get an error
    check = np.vectorize(lambda s : any(map(str.isdigit, s)))
    
    if isinstance(strs, list):
        strs = np.array(strs)
    
    # the indexed option allows us to filter out strings that don't contain digits.
    ## This prevents an error
    if indexed:
        strs = strs[check(strs)]

    #if indexed != True, then we don't filter the list of input strings and simply return it
    ##because an attempt to sort on indices (digits) that aren't present results in an error
    else:
        if not all(check(strs)):
            
            warnings.warn("Not all strings contain a number, returning unsorted input list to avoid throwing an error. "
                        "If you want to only consider strings that contain a digit, set indexed to True ")
            
            return strs
    
    get_index = np.vectorize(functools.partial(num_str, return_str=False, return_num=True))
    indices = get_index(strs).argsort()
    
    if max:
        return strs[np.flip(indices)]
    
    else:
        return strs[indices]

def lsdir(dir, keyword: "list or str" = None,
          exclude: "list or str" = None,
          indexed:bool=False):
    
    """ full path version of os.listdir with files/directories in order
    
        dir ::: path to a directory (str), required
        keyword ::: filter out strings that DO NOT contain this/these substrings (list or str)=None
        exclude ::: filter out strings that DO contain this/these substrings (list or str)=None
        indexed ::: filter out strings that do not contain digits.
                    Is passed to sort_strs function (bool)=False"""

    if dir[-1] == "/":
        dir = dir[:-1] # slicing out the final '/'

    listed_dir = os.listdir(dir) # list out the directory 

    if keyword is not None:
        listed_dir = keyword_strs(listed_dir, keyword) # return all items with keyword
    
    if exclude is not None:
        listed_dir = keyword_strs(listed_dir, keyword=exclude, exclude=True) # return all items without excluded str/list

    # Sorting (if possible) and ignoring hidden files that begin with "." or "~$"
    return [f"{dir}/{i}" for i in sort_strs(listed_dir, indexed=indexed) if (not i.startswith(".")) and (not i.startswith("~$"))] 

def keyword_strs(strs: list, keyword: "list or str", exclude: bool = False):
    
    if isinstance(keyword, str): # if the keyword is just a string 
        
        if exclude:
            filt = lambda string: keyword not in string
        
        else:
            filt = lambda string: keyword in string

    else:
        if exclude:
            filt = lambda string: all(kw not in string for kw in keyword)

        else:
            filt = lambda string: all(kw in string for kw in keyword)

    return list(filter(filt, strs))

def get_filename(filepath):
    """ returns a string of the file name of a filepath """
    return filepath.split('/')[-1].split('.')[0]

def chk_mkdir(newdir, nreturn=False):
    """ Checks and makes the directory if it doesn't exist. If True, nreturn will return the new file path. """
    isExist = os.path.exists(newdir)
    if not isExist:
        os.mkdir(newdir)
    if nreturn == True:
        return newdir

def get_filename(filepath):
    """ returns a string of the file name of a filepath """
    return filepath.split('/')[-1].split('.')[0]

def remove_numbers(arr, numbers):
    """
    Remove specific numbers from a NumPy array using numpy.delete().

    :param arr: NumPy array.
    :param numbers: List or array of numbers to remove.
    :return: NumPy array with specified numbers removed.
    """
    return np.delete(arr, np.where(np.isin(arr, numbers)))

def fix_protein_pdb(protein_path, outdir, bad_lines=np.array([13, 14])): 
    """ There unfortunately does not appear to be a good way to do this """
    with open(protein_path, 'r') as openfile: 
        lines = openfile.readlines()

    lines_arr = np.array(lines)
    idx = remove_numbers(np.arange(len(lines)), bad_lines)

    new_lines = lines_arr[idx]
    with open(f'{outdir}/fixed_protein.pdb', 'w+') as openfile: 
        for line in new_lines: 
            openfile.write(line)

    return f'{outdir}/fixed_protein.pdb'

def fix_ligand_pdb(ligand_path, outdir):
    with open(ligand_path, 'r') as openfile: 
        lines = openfile.readlines()

    lines_fixed = [line for line in lines if 'CONECT' not in line]

    with open(f'{outdir}/fixed_ligand.pdb', 'w+') as openfile: 
        for line in lines_fixed:
            openfile.write(line)

    return f'{outdir}/fixed_ligand.pdb' 

def make_tleapin(outdir, protein_path, ligand_path): 
    with open(f'{outdir}/tleap.in', 'w+') as open_file: 
        # writing in the force fields we'll need
        open_file.write('source leaprc.protein.ff14SB \nsource leaprc.gaff \nloadamberparams frcmod.ionsjc_tip3p \n')

        # loading the pdbs 
        open_file.write(f'lig = loadmol2 {ligand_path} \nrec = loadpdb {protein_path} \n')
        open_file.write('complex = combine {rec lig} \n')

        # saving the files we need 
        open_file.write(f'saveamberparm lig {outdir}/ligand.prmtop {outdir}/ligand.inpcrd \n')
        open_file.write(f'saveamberparm rec {outdir}/receptor.prmtop {outdir}/receptor.inpcrd \n')
        open_file.write(f'saveamberparm complex {outdir}/complex.prmtop {outdir}/complex.inpcrd \n')

        # and then mother fucking quit this bitch! 
        open_file.write('quit')
    return f'{outdir}/tleap.in'

def make_mmgbsain(outdir): 
    with open(f'{outdir}/mmgbsa.in', 'w+') as openfile: 
        openfile.write('&general \n')
        openfile.write('    interval=1,\n')
        openfile.write('    verbose=2,\n')
        openfile.write('/ \n')
        openfile.write('&gb \n')
        openfile.write('/ \n')
    
    return f'{outdir}/mmgbsa.in'

def parse_mmpdsa_dat(dat_file): 
    with open(dat_file, 'r') as openfile: 
        lines = openfile.readlines()
    
    return float(lines[-5].split()[2])




def run_mmgbsa(outdir, traj, pdb, lig_resid=20, nc='1'): 
    """ You'll need to change the nc to 0 for ligand 23. """

    # First, directories: 
    traj_name = get_filename(traj)
    traj_outdir = chk_mkdir(f'{outdir}/{traj_name}_out', nreturn=True)
    energy_dir = chk_mkdir(f'{outdir}/mmgbsa_energies', nreturn=True)

    # Then, loading: 
    trajectory = md.load(traj, top=pdb, stride=1)
    protein = trajectory.atom_slice(trajectory.top.select('protein'))
    ligand = trajectory.atom_slice(trajectory.top.select(f'resid {lig_resid}'))

    energies = []

    # Now, doing our thing for all the frames: 
    for frame in range(trajectory.n_frames): 
        # first, creating a directory for this frame 
        frame_dir = chk_mkdir(f'{traj_outdir}/c_{frame}_out', nreturn=True)
        print(outdir)
        print(traj_outdir)
        print(frame_dir)
        # working on the protein: 
        protein[frame].save_pdb(f'{frame_dir}/protein.pdb')
        fixed_protein_pdb = fix_protein_pdb(f'{frame_dir}/protein.pdb', frame_dir)
        sp.run(['pdb4amber', '-i', fixed_protein_pdb, '-o' ,f'{frame_dir}/protein4amber.pdb', '--nohyd', '--add-missing-atoms'])

        # working on the ligand now: 
        ligand[frame].save_pdb(f'{frame_dir}/ligand.pdb')
        fixed_ligand_pdb = fix_ligand_pdb(f'{frame_dir}/ligand.pdb', frame_dir)
        sp.run(['antechamber', '-i', fixed_ligand_pdb, '-fi', 'pdb','-o' ,f'{frame_dir}/lig.mol2', '-fo', 'mol2', '-c', 'bcc', '-nc', nc ])
        
        # Now, creating the tleap script so we can get our prmtop and incprd files: 
        tleap_file = make_tleapin(frame_dir, f'{frame_dir}/protein4amber.pdb',f'{frame_dir}/lig.mol2')
        sp.run(['tleap', '-s', '-f', tleap_file])

        # then, making the complex pdb from our files: 
        complex_pdb = sp.run(['ambpdb', '-p', f'{frame_dir}/complex.prmtop', '-c', f'{frame_dir}/complex.inpcrd'], capture_output=True)
        with open(f'{frame_dir}/complex.pdb', 'w+') as openfile: 
            openfile.write(complex_pdb.stdout.decode('utf-8'))

        # finally, running mmgbsa with all of our files: 
        mmgbsa_file = make_mmgbsain(frame_dir)
        sp.run(['MMPBSA.py', '-O', '-i', mmgbsa_file, '-o', f'{frame_dir}/FINAL_RESULTS_MMPBSA.dat', '-sp', f'{frame_dir}/complex.prmtop', '-cp', 
                 f'{frame_dir}/complex.prmtop', '-rp', f'{frame_dir}/receptor.prmtop', '-lp', f'{frame_dir}/ligand.prmtop', '-y', f'{frame_dir}/complex.pdb'])

        # then, parsing to get our energy: 
        energy = parse_mmpdsa_dat(f'{frame_dir}/FINAL_RESULTS_MMPBSA.dat')
        energies.append(energy)
    
    np.save(f'{energy_dir}/{traj_name}_mmgbsa_energies.npy', np.array(energies))




if __name__ == '__main__': 


##### ARGPARSE ######
    parser = argparse.ArgumentParser(description = "Generates mmgbsa energies for a trajectory and topology file.")
    
    parser.add_argument("--outdir", required=True, type=str, help="The directory where you'd like to save all of your data!")

    parser.add_argument("--traj", required = True, type = str, help = "The trajectory ")

    parser.add_argument("--pdb", required = True, type = str, help = "The pdb to load the trajectory ")

    args = parser.parse_args()
################


    run_mmgbsa(args.outdir, args.traj, args.pdb)
