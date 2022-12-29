# class DogeFold(object):
def grad_with_xtc( model_weights, train_data, test_data, nstep):
    '''
    - flowchart
    - load model to gpu
    - evaluate on test sets
        - calculate gradient for samples in traj
        - loss function with FAPE? (fape is great to avoid alignment)
        - dogefold.grad_with_xtc( model_weights, train_data, test_data, nstep)
    - load weights from pointer
        - parse xtc trajectories into paired frames of backbone coords
        - function to fold back 
    '''

# dogefold.grad_with_xtc( model_weights, train_data, test_data, nstep)


import MDAnalysis

from gcat_util import AminoAcidResidue
aa_by_trip = AminoAcidResidue.aa_by_trip
import torch
import torch.nn as nn
import numpy as np


def extract_traj_0(DIR,device):
    if device is None:
        device = torch.device('cpu')
    fn  = f'{DIR}/md_0_1.xtc'

    xu = u = MDAnalysis.Universe(f'{DIR}/md_0_1.gro', f'{DIR}/md_0_1.xtc')

    bb = u.select_atoms('protein and backbone and name CA')  # a selection (AtomGroup)

    bb0 = bb.copy()[0]
    res_list = bb.resnames
    bb_coords = []
    for ts in u.trajectory:
        bb_coords.append(bb.positions)

    bb_coords = np.stack(bb_coords,0)
    bb_coords = torch.tensor(bb_coords, device=device)

    res_list = [aa_by_trip[aa][0] for aa in res_list]
    res_list = torch.tensor([res_list],device =device)

    return res_list, bb_coords

class MyModel(nn.Module):
    def __init__(self,config,device,):
        super().__init__()

        self.config = config
        self.device = device
        n_vocab     = config['n_vocab']
        embed_dim   = config['embed_dim']
        kernel_size = config['kernel_size']
        n_layer     = config['n_layer']
        E           = embed_dim
        C           = config['comp_size']
        self.step_size = config['step_size']

        self.emb = nn.Embedding(n_vocab, embed_dim).to(self.device)
        self.layers = nn.ModuleList([nn.Conv1d(embed_dim,embed_dim,kernel_size=kernel_size, padding='same')
            for _ in range(n_layer)]).to(self.device)

        # x= nn.Linear(E,E*C) 
        # self.t1 = nn.Parameter(x.weight.reshape((E,E,C))).to(self.device)

        x= nn.Linear(E,E*C)
        self.t1 = x.to(self.device)

        x= nn.Linear(3,3*C)
        self.t2 = nn.Parameter(x.weight.reshape((3,3,C))).to(self.device)

        x= nn.Linear(3,3*C)
        self.t2 = x.to(self.device)
        
    def call(self,x):
        x = self.emb(x)
        x = x.transpose(2,1)
        # print(x.shape)
        for layer in self.layers:
            x = (x + layer(x)).tanh()
            # x = 0.5*(x + layer(x).tanh())
        x = x.transpose(2,1)
        return x

    def get_loss(model, aa_array:torch.Tensor, bb_coords: torch.Tensor):
        '''
        aa_array: tensor of shape (Batch, Length, ) indicating residue feature
        bb_coords: tensor of shape (Batch, Length, 3) indicating
           torch.Size([1, 56])
           torch.Size([101, 56, 3])
        '''
        self = model
        
        x  = ret =   model.call(aa_array)
        xx = bb_coords[:,:,None] - bb_coords[:,None,:]
        xx = torch.tensor(xx)   
        # proposal_vects = xxd = torch.tensordot(xx,self.t2,1)
        sp = list(xx.shape)
        C = self.config['comp_size']
        proposal_vects = self.t2(xx).reshape(sp+[C])

        sp = list(x.shape)
        x2 = self.t1(x).reshape(sp+[C])
        # x2 = torch.tensordot(x,self.t1,1)
        xa = x[:,None].repeat((1,C,1,1))
        xb = x2.transpose(1,3)
        xc = xa.matmul(xb)

        ### average of proposal_vects
        vects_disp= proposal_vects.matmul( xc.softmax(1).transpose(1,-1).unsqueeze(-1)).squeeze(-1).sum(2)

        # new_bb_coords = bb_coords + 0.1*vects_disp
        new_bb_coords = bb_coords + self.step_size*vects_disp
        # new_bb_coords = bb_coords + 0.001*vects_disp

        mse = (new_bb_coords[1:] - bb_coords[:-1]).square().sum(-1).mean(1)
        return mse


def dmpfold2_aln_to_coords(
        input_file, device, template=None, iterations=default_iterations,
                  minsteps=default_minsteps, weights_file=None, return_alnmat=False):
    device = torch.device(device)

    # Create neural network model (depending on first command line parameter)
    network = GRUResNet(512,128).eval().to(device)

    modeldir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'trained_model')

    if weights_file is None:
        # Use standard weights
        if not os.path.isfile(os.path.join(modeldir, 'FINAL_fullmap_e2e_model_part1.pt')):
            download_trained_model(modeldir)

        # Model parameters stored as two files to get round GitHub's file size limit
        trained_model = torch.load(os.path.join(modeldir, 'FINAL_fullmap_e2e_model_part1.pt'),
                                    map_location=lambda storage, loc: storage)
        trained_model.update(torch.load(os.path.join(modeldir, 'FINAL_fullmap_e2e_model_part2.pt'),
                                    map_location=lambda storage, loc: storage))
        
    else:
        # Use weights specified in file
        trained_model = torch.load(weights_file, map_location=lambda storage, loc: storage)

    network.load_state_dict(trained_model)

    aln = []
    with open(input_file, 'r') as alnfile:
        for line in alnfile.readlines():
            if not line.startswith(">"):
                aln.append(line.rstrip())

    if template is not None:
        with open(template, 'r') as tpltpdbfile:
            coords = []
            n = 0
            for line in tpltpdbfile:
                if line[:4] == 'ATOM' and line[12:16] == ' CA ':
                    # Split the line
                    pdb_fields = [line[:6], line[6:11], line[12:16], line[17:20], line[21],
                                  line[22:26], line[30:38], line[38:46], line[46:54]]
                    coords.append(np.array([float(pdb_fields[6]), float(pdb_fields[7]), float(pdb_fields[8])], dtype=np.float32))

        init_coords = torch.from_numpy(np.asarray(coords)).unsqueeze(0).to(device)
    else:
        init_coords = None

    nloops = max(iterations, 0)
    refine_steps = max(minsteps, 0)

    aa_trans = str.maketrans('ARNDCQEGHILKMFPSTWYVBJOUXZ-.', 'ABCDEFGHIJKLMNOPQRSTUUUUUUVV')

    nseqs = len(aln)
    length = len(aln[0])
    alnmat = (np.frombuffer(''.join(aln).translate(aa_trans).encode('latin-1'), dtype=np.uint8) - ord('A')).reshape(nseqs,length)

    if nseqs > 3000:
        alnmat = alnmat[:3000]
        nseqs = 3000

    inputs = torch.from_numpy(alnmat).type(torch.LongTensor).to(device)

    msa1hot = F.one_hot(torch.clamp(inputs, max=20), 21).float()
    w = reweight(msa1hot, cutoff=0.8)

    f2d_dca = fast_dca(msa1hot, w).float() if nseqs > 1 else torch.zeros((length, length, 442), device=device) 
    f2d_dca = f2d_dca.permute(2,0,1).unsqueeze(0)

    if init_coords is not None:
        dmap = (init_coords - init_coords.transpose(0,1)).pow(2).sum(dim=2).sqrt().unsqueeze(0).unsqueeze(0)
    else:
        dmap = torch.zeros((1, 1, length, length), device=device) - 1

    inputs2 = torch.cat((f2d_dca, dmap), dim=1)

    network.eval()
    with torch.no_grad():
        coords, confs = network(inputs, inputs2, nloops, refine_steps)
        coords = coords.view(-1,length,5,3)[0]
        confs = confs[0]

    if return_alnmat:
        return coords, confs, alnmat
    else:
        return coords, confs


# torch.save(model.state_dict(), filepath)
# model.load_state_dict(torch.load(filepath))

import torch.optim as optim
def main():
    # device = torch.device('cpu')
    device = torch.device('cuda:0')


    data = {}
    DIR = '/shared_data/p_md_simple_pull/3320a3e0fd034275af677e6541702d1a'
    aa_array, bb_coords = extract_traj_0( DIR, device)
    data['train'] =  (aa_array, bb_coords)
    # data['test'] =  (aa_array, bb_coords)
    # DIR = '/shared_data/p_md_simple_pull/f157c8c9e0d3888d73edcb9e1f484945'
    DIR = '/shared_data/p_md_simple_pull/2dc95c69e402e918094fab1314b673a2'
    data['test'] = extract_traj_0( DIR, device)
    # data['train'] = extract_traj_0( DIR, device)

#    2dc95c69e402e918094fab1314b673a2
    
    'http://www.cathdb.info/version/v4_3_0/api/rest/id/1hdoA00.pdb'

    model = MyModel(dict(embed_dim=20,n_vocab=21,kernel_size=7,n_layer = 11, comp_size=20,
    # step_size=0.01,
    step_size=0.0001,
    # step_size=0.0000,
    
    ),device=device)

    bb_coords = torch.tensor(bb_coords,device=device)
    aa_array  = torch.tensor(aa_array,device=device)

    lr = 0.0005
    n_steps = 1200
    #### Training loop

    opt = optim.RMSprop(model.parameters(),lr=lr)
    for i in range(n_steps):
        
        print()
        for k in 'train test'.split():
            aa_array,bb_coords = data[k]
            mse = model.get_loss(aa_array, bb_coords)
            # mse = mse.mean(0)
            mse = mse.mean(0)
            print(f'[MSE,{k}]{mse:.3f}',end='    ')
        # print()
            if k=='train':
                mse.backward()
                opt.step()
main()