print("hello. nue200")
import sys
import yaml
import numpy as np
import csv
sys.path.append('/sdf/group/neutrino/yjwa/lartpc_mlreco3d')
from mlreco.main_funcs import process_config, prepare
from mlreco.utils.deghosting import adapt_labels_knn
from scipy.special import softmax
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_label

from numba import njit
from numba.typed import List

# Define the data loader configuration
#from notebook_utils import get_inference_cfg
#cfg = yaml.safe_load(open('/sdf/group/neutrino/drielsma/inference_full_volume_100522.cfg')) # Includes interaction clustering
#cfg = yaml.safe_load(open('/sdf/group/neutrino/yjwa/cfgs/inference_full_volume_100522.cfg'))
cfg = yaml.safe_load(open('/sdf/group/neutrino/yjwa/cfgs/inference_interGrapPA_volume_110722.cfg'))
# cfg = get_inference_cfg(cfg_path = '/sdf/group/neutrino/drielsma/me/train/icarus_newk/grappa_inter_label_mpr_primary.cfg', batch_size=1) # Includes interaction clustering
#cfg['iotool']['dataset']['data_keys'] = ['/sdf/group/neutrino/ldomine/mpvmpr_062022_v1[1,2]/*.root'] # MPV/MPR
#cfg['iotool']['dataset']['data_keys'] = ['/sdf/group/neutrino/ldomine/mpvmpr_062022_v11/larcv_0001.root'] # single MPV/MPR
#cfg['iotool']['dataset']['data_keys'] = ['/sdf/group/neutrino/ldomine/nue_052022_v0[4,6,7]/all_filtered.root'] # NUE
#cfg['iotool']['dataset']['data_keys'] = ['/sdf/group/neutrino/ldomine/nue_052022_v06/all_filtered.root'] # NUE
cfg['iotool']['dataset']['data_keys'] = ['/sdf/group/neutrino/ldomine/numu_052022_v0[5,6]/all_filtered.root'] # NUMU

# Pre-process configuration
process_config(cfg, verbose=False)
# Instantiate "handlers
hs = prepare(cfg)#, event_list=[76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86 ,87])

with open ('numu_feats_kin_dense_all.csv', 'w') as f:
    w_MPVMPR = csv.writer(f)
    w_MPVMPR.writerow(['idx', 'is_nu', 'is_primary', 'group_id', 'int_id', 'px', 'py', 'pz', 'p', 'E_init', 'pdg', 'parent_pdg','primary_score', 'type_label', 'type_pred', 'first_step', 'last_step'])
    for iteration in range(len(hs.data_io)):
        print(iteration)
        event, output = hs.trainer.forward(hs.data_io_iter)
        #print("2")
        cluster_label = event['cluster_label'][0]
        kinematics_label = event['kinematics_label'][0]
        #cluster_label = adapt_labels_knn(output, event['segment_label'], event['cluster_label'], use_numpy=True)[0]
        #kinematics_label = adapt_labels_knn(output, event['segment_label'], event['kinematics_label'], use_numpy=True)[0]
        particles = event['particles'][0]
        #node_feats = output['input_node_features'][0]
        if ('input_node_features' not in output.keys()):
            node_feats = [-1]
        #    print("pixels", len(output['input_rescaled'][0]))
        #    continue
        else:
            node_feats = output['input_node_features'][0]
        #print(np.shape(kinematics_label))
        #print(np.unique(kinematics_label[:,5]))
        input_particles = form_clusters(event['input_data'][0], column=6)

        if( (np.unique(kinematics_label[:,5])==[-1.]).all()):
            continue
        primary_labels  = get_cluster_label(kinematics_label, input_particles, column=12)
        interaction_labels = get_cluster_label(cluster_label, input_particles, column=7)
        nu_id_labels = get_cluster_label(cluster_label, input_particles, column=8)
        type_labels  = get_cluster_label(cluster_label, input_particles, column=9)
        group_ids = get_cluster_label(cluster_label, input_particles, column=6)
        type_preds = np.argmax(output['node_pred_type'][0], axis=1)
        primary_preds = softmax(output['node_pred_vtx'][0][:,-2:], axis=1)[:,-1]
        for i in range(len(nu_id_labels)):
            #if (nu_id_labels[i]==1): # mpv only 
            px=particles[int(group_ids[i])].px()
            py=particles[int(group_ids[i])].py()
            pz=particles[int(group_ids[i])].pz()
            pdg=particles[int(group_ids[i])].pdg_code()
            parent=particles[int(group_ids[i])].parent_pdg_code()
            p=particles[int(group_ids[i])].p()
            E_init=particles[int(group_ids[i])].energy_init()
            first_step=particles[int(group_ids[i])].first_step()
            last_step=particles[int(group_ids[i])].last_step()
            row = [iteration, nu_id_labels[i], primary_labels[i], group_ids[i], interaction_labels[i], px, py, pz, p, E_init, pdg, parent, primary_preds[i], type_labels[i], type_preds[i], first_step.x(), first_step.y(), first_step.z(), last_step.x(), last_step.y(), last_step.z()]
            row = np.concatenate([row, node_feats[i]])
                #print (row)
            w_MPVMPR.writerow(row)
        #print("3")
    print("done.")        
