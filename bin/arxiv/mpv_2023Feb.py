import sys
import yaml
import numpy as np
import csv
sys.path.append('/sdf/group/neutrino/yjwa/lartpc_mlreco3d')
from mlreco.main_funcs import process_config, prepare
from mlreco.utils.deghosting import adapt_labels_knn
from scipy.special import softmax
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_label

# Define the data loader configuration
#from notebook_utils import get_inference_cfg
cfg = yaml.safe_load(open('/sdf/group/neutrino/yjwa/cfgs/grappa_inter_label_nompr_val_particle.cfg'))
cfg['iotool']['dataset']['data_keys'] = ['/sdf/group/neutrino/ldomine/mpvmpr_112022_v11/all_*.root'] # localized mpv
cfg['iotool']['batch_size'] = 1 
#cfg['trainval']['model_path'] = '/sdf/group/neutrino/drielsma/me/train/icarus_localized/weights/grappa_inter/label_nompr/snapshot-94999.ckpt'

cfg['trainval']['model_path'] = '/sdf/group/neutrino/yjwa/train/icarus_localized/weights/grappa_inter/label_nompr/snapshot-96499.ckpt'  
#cfg['iotool']['dataset']['schema']['input_data']['args']['sparse_value_event_list'] = ['sparse3d_reco_hit_charge0', 'sparse3d_reco_hit_charge1', 'sparse3d_reco_hit_charge2', 'sparse3d_reco_hit_key0', 'sparse3d_reco_hit_key1', 'sparse3d_reco_hit_key2', 'sparse3d_pcluster_semantics_ghost']
# Pre-process configuration
process_config(cfg, verbose=False)
hs = prepare(cfg)

print(len(hs.data_io))

with open ('mpvmpr_kin_study_model96499.csv', 'w') as f:
    w_MPVMPR = csv.writer(f)
    for iteration in range(len(hs.data_io)):
        print(iteration)
        event, output = hs.trainer.forward(hs.data_io_iter)
        #print("1")
        clusts = output['clusts'][0]
        particles = event['particles'][0]
        #cluster_label = event['cluster_label'][0]
        #print("2")
        group_id = get_cluster_label(event['input_data'][0], clusts, column=6)
        input_nu_id = get_cluster_label(event['input_data'][0], clusts, column=8)
        #print("nu_id:", input_nu_id)
        input_type = get_cluster_label(event['input_data'][0], clusts, column=9)
        #print("type:", input_type)
        input_pri_id = get_cluster_label(event['input_data'][0], clusts, column=15)
        #print("primaryID:", input_pri_id)
        type_preds = np.argmax(output['node_pred_type'][0], axis=1)
        #print("type_preds: ", type_preds)
        #print("type: ", output['node_pred_type'][0])
        types = softmax(output['node_pred_type'][0],axis=1)
        primary_preds = softmax(output['node_pred_vtx'][0][:,-2:], axis=1)[:,-1]
        #print("primary_preds: ", primary_preds)
        #print("len_input: ", len(input_nu_id), " , len_pred: ", len(primary_preds))
        

        if(len(input_nu_id)>0):
            for i in range(len(input_nu_id)):
                temp_part = particles[int(group_id[i])]
                pdg = temp_part.pdg_code()
                e_init = temp_part.energy_init()
                e_depo = temp_part.energy_deposit()
                num_pix = len(clusts[i])
                px = temp_part.px()
                py = temp_part.py()
                pz = temp_part.pz()
                row = [iteration, input_nu_id[i], input_pri_id[i], primary_preds[i], input_type[i], type_preds[i], types[i][0], types[i][1], types[i][2], types[i][3], types[i][4], group_id[i], pdg, e_init, e_depo, px, py, pz, num_pix]
                #print(row)
#, input_type[i], type_preds[i], types[i][0], types[i][1], types[i][2], types[i][3], types[i][4]]
                w_MPVMPR.writerow(row)
        #print("3")
    print("done.")        
