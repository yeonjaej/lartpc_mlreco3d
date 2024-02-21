import sys
import yaml
import numpy as np
import csv
sys.path.append('/sdf/group/neutrino/yjwa/lartpc_mlreco3d')
from mlreco.main_funcs import process_config, prepare
from mlreco.utils.deghosting import adapt_labels_knn
from scipy.special import softmax
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_label

cfg = yaml.safe_load(open('/sdf/group/neutrino/yjwa/cfgs/grappa_inter_label_nompr_val_particle.cfg'))

cfg['iotool']['dataset']['data_keys'] = ['/sdf/group/neutrino/ldomine/mpvmpr_112022_v11/all_0.root']# localized mpv
cfg['iotool']['batch_size'] = 1                                                          
cfg['trainval']['model_path'] = '/sdf/group/neutrino/drielsma/me/train/icarus_localized/weights/grappa_inter/label_nompr/snapshot-94999.ckpt'

process_config(cfg, verbose=False)
# Instantiate "handlers
hs = prepare(cfg)#, event_list=[122])

import ROOT as ROOT
from ROOT import TFile, TChain
example = TFile('/sdf/group/neutrino/ldomine/mpvmpr_112022_v11/all_0.root')
example.ls()
particle_mpv_tree = example.Get("particle_mpv_tree")

trackid_list = []

for entry in range(particle_mpv_tree.GetEntries()):
    particle_mpv_tree.GetEntry(entry)
    #event = particle_mpv_tree.particle_mpv_branch
    trackid = particle_mpv_tree.GetLeaf("_part_v._trackid")
    temp_trackid_list = []
    for i in range(trackid.GetLen()):
        temp_trackid_list.append(trackid.GetValue(i)+1)
    trackid_list.append(temp_trackid_list)



with open ('mpv_select_kin.csv', 'w') as f:
    w_MPVMPR = csv.writer(f)
    #w_MPVMPR.writerow(['idx', 'is_nu', 'is_primary', 'group_id', 'int_id', 'px', 'py', 'pz', 'p', 'E_init', 'pdg', 'parent_pdg','primary_score', 'type_label', 'type_pred', 'first_step', 'last_step'])
    
    for iteration in range(len(hs.data_io)):
        interaction_id = -1
        #if (int(iteration)==122):
        #    continue
        print(iteration)
        event, output = hs.trainer.forward(hs.data_io_iter)
        
        # find mpv interaction id
        particles = event['particles'][0]
        mpv_groupid_list=[]
        for i in range (len(particles)):
            #print(part.track_id())
            if (particles[i].track_id() in trackid_list[iteration]):
                mpv_groupid_list.append(i)
                print("group id : ", i)

        primary_labels  = get_cluster_label(event['input_data'][0], output['clusts'][0], column=15)
        interaction_labels = get_cluster_label(event['input_data'][0], output['clusts'][0], column=7)
        nu_id_labels = get_cluster_label(event['input_data'][0], output['clusts'][0], column=8)
        type_labels  = get_cluster_label(event['input_data'][0], output['clusts'][0], column=9)
        group_ids = get_cluster_label(event['input_data'][0], output['clusts'][0], column=6)

        type_preds = np.argmax(output['node_pred_type'][0], axis=1)
        #print("type_preds: ", type_preds)                                                                                                                            
        
        types = softmax(output['node_pred_type'][0],axis=1)
        primary_preds = softmax(output['node_pred_vtx'][0][:,-2:], axis=1)[:,-1]
        #print("primary_preds: ", primary_preds)                                                                                                                      
        #print("len_input: ", len(input_nu_id), " , len_pred: ", len(primary_preds))                                                                                  
        if(len(interaction_labels)>0):
            for i in range(len(interaction_labels)):
                if (group_ids[i] in mpv_groupid_list): # mpv only     
            
                    row = [iteration, nu_id_labels[i], group_ids[i], primary_labels[i], primary_preds[i], type_labels[i], type_preds[i], types[i][0], types[i][1], types[i][2], types[i][3], types[i][4]]
                    print(row)
                    w_MPVMPR.writerow(row)
        #print("3")                                                             


    print("done.")        
