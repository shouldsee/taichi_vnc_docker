# idx = [-1]
# lst = []
class AminoAcidResidue(object): 
    idx = [-1]
    lst = []
    # lst = lst    
    # idx = [-1]
    def __init__( self, full, trip, letter,alt=None):
        self.idx[0] += 1
        self.lst.append((self.idx[0],full,trip,letter))
        pass
ALA = AminoAcidResidue('Alanine', 'ALA', 'A')
ARG = AminoAcidResidue('Arginine', 'ARG', 'R')
ASN = AminoAcidResidue('Asparagine', 'ASN', 'N')
ASP = AminoAcidResidue('Aspartate' ,'ASP', 'D', ['ASH', 'AS4'])
CYS = AminoAcidResidue('Cysteine', 'CYS', 'C', ['CYM', 'CYX'])
GLU = AminoAcidResidue('Glutamate', 'GLU', 'E', ['GLH', 'GL4'])
GLN = AminoAcidResidue('Glutamine', 'GLN', 'Q')
GLY = AminoAcidResidue('Glycine', 'GLY', 'G')
HIS = AminoAcidResidue('Histidine', 'HIS', 'H', ['HIP', 'HIE', 'HID'])
HYP = AminoAcidResidue('Hydroxyproline', 'HYP', None)
ILE = AminoAcidResidue('Isoleucine', 'ILE', 'I')
LEU = AminoAcidResidue('Leucine', 'LEU', 'L')
LYS = AminoAcidResidue('Lysine', 'LYS', 'K', ['LYN'])
MET = AminoAcidResidue('Methionine', 'MET', 'M')
PHE = AminoAcidResidue('Phenylalanine', 'PHE', 'F')
PRO = AminoAcidResidue('Proline', 'PRO', 'P')
SER = AminoAcidResidue('Serine', 'SER', 'S')
THR = AminoAcidResidue('Threonine', 'THR', 'T')
TRP = AminoAcidResidue('Tryptophan', 'TRP', 'W')
TYR = AminoAcidResidue('Tyrosine', 'TYR', 'Y')
VAL = AminoAcidResidue('Valine', 'VAL', 'V')
AminoAcidResidue.aa_by_trip = {xx[2]:xx for xx in AminoAcidResidue.lst}