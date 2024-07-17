from collections import OrderedDict
import numpy as np

from analysis.producers.decorator import write_to
from analysis.classes.data import *
from analysis.producers.logger import ParticleLogger, InteractionLogger

attached_threshold = 3 # cm
ablation_eps = 6 # cm
ablation_radius = 4.5 # cm
ablation_min_samples = 5

from analysis.post_processing.evaluation.match import generate_match_pairs
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=ablation_eps, min_samples=ablation_min_samples)
from collections import OrderedDict

# Michel import
from mlreco.utils.globals import TRACK_SHP, MICHL_SHP

is_data = False

muon_min_voxel_count = 20
#matching_mode = "true_to_pred"
shower_threshold = 10
shower_label = 0

from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

def is_attached_at_edge(points1, points2, attached_threshold=3,
                        one_pixel=ablation_eps, ablation_radius=4.5, ablation_min_samples=5,
                        return_dbscan_cluster_count=False):
    distances = cdist(points1, points2)
    is_attached = np.min(distances) < attached_threshold
    # check for the edge now
    Michel_min, MIP_min = np.unravel_index(np.argmin(distances), distances.shape)
    min_coords = points2[MIP_min, :]
    ablated_cluster = points2[np.linalg.norm(points2-min_coords, axis=1) > ablation_radius]
    new_cluster_count, old_cluster_count = 0, 1
    if ablated_cluster.shape[0] > 0:
        dbscan = DBSCAN(eps=one_pixel, min_samples=ablation_min_samples)
        old_cluster = dbscan.fit(points2).labels_
        new_cluster = dbscan.fit(ablated_cluster).labels_
        # If only one cluster is left, we were at the edge
        # Account for old cluster count in case track is fragmented
        # and put together by Track GNN
        old_cluster_count = len(np.unique(old_cluster[old_cluster>-1]))
        new_cluster_count =  len(np.unique(new_cluster[new_cluster>-1]))
        is_edge = (old_cluster_count - new_cluster_count) <= 1 and old_cluster_count >= new_cluster_count
    else: # if nothing is left after ablating, this is a really small muon... calling it the edge
        is_edge = True

    if return_dbscan_cluster_count:
        return is_attached and is_edge, new_cluster_count, old_cluster_count

    return is_attached and is_edge


@write_to(['michels', 'true_michels'])
def michel(data_blob, res, **kwargs):
    """
    Template for a logging script for particle and interaction objects.

    Parameters
    ----------
    data_blob: dict
        Data dictionary after both model forwarding post-processing
    res: dict
        Result dictionary after both model forwarding and post-processing

    Returns
    -------
    interactions: List[List[dict]]
        List of list of dicts, with length batch_size in the top level
        and length num_interactions (max between true and reco) in the second

    particles: List[List[dict]]
        List of list of dicts, with same structure as <interactions> but with
        per-particle information.

    Information in <interactions> will be saved to $log_dir/interactions.csv
    and <particles> to $log_dir/particles.csv.
    """

    michels, true_michels = [], []
    is_data = False
    
    
            
        #matching_mode         = kwargs['matching_mode']
        #particle_fieldnames   = kwargs['logger'].get('particles', {})
        #int_fieldnames        = kwargs['logger'].get('interactions', {})

    image_idxs = data_blob['index']
    #    meta       = data_blob['meta'][0]

    for idx, index in enumerate(image_idxs):

        index_dict = {
                'Index': index,
                'file_index': data_blob['file_index'][idx]
        }
        particles = res['particles'][idx]
        #truth_particles = res['truth_particles'][idx]
    
        if not is_data:
            truth_particles = res['truth_particles'][0]
                
            matched_particles = generate_match_pairs(truth_particles, particles)
            matched_r2t = matched_particles['matches_r2t']
            matched_t2r = matched_particles['matches_t2r'] 
            for tp in truth_particles:
                if tp.semantic_type != MICHL_SHP: continue
                if not tp.is_contained: continue
            
                michel_is_attached_at_edge = False
                for tp2 in truth_particles:
                    if tp2.semantic_type != TRACK_SHP: continue
                    if tp2.size < muon_min_voxel_count: continue
                
                    attached_at_edge= is_attached_at_edge(tp.truth_points, tp2.truth_points,
                                                      attached_threshold=attached_threshold,
                                                      #ablation_eps=ablation_eps,
                                    ablation_radius=ablation_radius,
                                                      ablation_min_samples=ablation_min_samples)
                
                    if not attached_at_edge: continue
                    michel_is_attached_at_edge = True
                    break

                true_michels.append(OrderedDict({
                    'index': index,
                    'is_attached_at_edge': michel_is_attached_at_edge,
                #'has_reco': has_reco,
                #'matched': is_matched,
                    'true_num_pix': tp.size,
                'true_sum_pix': tp.depositions.sum(),
                "true_particle_energy_deposit": tp.energy_deposit,
                'true_semantic_type': tp.semantic_type,
                'true_particle_pdg': tp.pdg_code,
                'true_particle_parent_pdg': tp.parent_pdg_code,
                "true_noghost_num_pix": tp.truth_points.shape[0],
                "true_noghost_sum_pix": tp.truth_depositions.sum(),
                "true_noghost_MeV_sum_pix": tp.truth_depositions_MeV.sum(),

                    #'distance_to_muon': distance_to_muon,
                #'michel_cluster_count': len(np.unique(michel_clusters[michel_clusters>-1])),
                #'volume': tp.volume
                }))


                #N_true_michel += 1#np.count_nonzero([tp.semantic_type == MICHL_SHP for tp in truth_particles_contained])
            # true_ids = np.array([p.id for p in truth_particles])


            
        for p in particles:
            muon = None
            if p.semantic_type != MICHL_SHP: continue
            if len(p.points)<20: continue
            if not p.is_contained: continue

            # Check whether it is attached to the edge of a track
            michel_is_attached_at_edge = False
            for p2 in particles:
                if p2.semantic_type != TRACK_SHP: continue
                if not is_attached_at_edge(p.points, p2.points,
                                           attached_threshold=attached_threshold,
                                           #ablation_eps=ablation_eps,
                                        ablation_radius=ablation_radius,
                                           ablation_min_samples=ablation_min_samples): continue
                michel_is_attached_at_edge = True
                muon = p2
                break
            # Require that the Michel is attached at the edge of a track
            if not michel_is_attached_at_edge: continue

            # Look for shower-like particles nearby
            #daughter_showers = []
            #distance_to_closest_shower = np.infty
            #if shower_threshold > -1:
            #    for p2 in particles:
            #        if p2.id == p.id: continue
            #        if p2.semantic_type != shower_label and p2.semantic_type != MICHL_SHP: continue
            #        distance_to_closest_shower = min(distance_to_closest_shower, cdist(p.points, p2.points).min())
            #        if distance_to_closest_shower >= shower_threshold: continue
            #        daughter_showers.append(p2)
            #daughter_num_pix = np.sum([daugher.size for daugher in daughter_showers])
            #daughter_sum_pix = np.sum([daughter.depositions.sum() for daughter in daughter_showers])

                # Record muon endpoint
            endpoint = np.array([-1, -1, -1])
            if cdist(p.points, [muon.start_point]).min() > cdist(p.points, [muon.end_point]).min():
                if (muon.end_point != np.array([-1, -1, -1])).any():
                    endpoint = muon.end_point
            else:
                if (muon.start_point != np.array([-1, -1, -1])).any():
                    endpoint = muon.start_point
                    #if (endpoint == np.array([-1, -1, -1])).all():
            endpoint_id = cdist(p.points, muon.points).argmin()
            michel_id, muon_id = np.unravel_index(endpoint_id, (len(p.points), len(muon.points)))
            endpoint = muon.points[muon_id]

            dbscan = DBSCAN(eps=ablation_eps, min_samples=ablation_min_samples)
            clabels = dbscan.fit(p.points).labels_
            pionization = clabels[michel_id] # cluster label of Michel point closest to the muon
            primary_ionization = clabels == pionization

            # Record distance to muon
            #michel_to_muon_distance = cdist(p.points, muon.points).min()

            # Record calibrated depositions sum
            #pred_sum_pix_calib = yz_calibrations(yz_calib, p.points, p.depositions)
            #pred_primary_sum_pix_calib = yz_calibrations(yz_calib, p.points[primary_ionization], p.depositions[primary_ionization])
            #print("caibrating ", p.depositions.sum(), pred_sum_pix_calib)

            # Record candidate Michel
            update_dict={
                'index': index,
                'volume': p.volume_id,
                'matched': False,
                "pred_num_pix": p.size,
                "pred_sum_pix": p.depositions.sum(), # p.calo_ke() or multiply 1/0.87 sub-optimal
                #"daughter_num_pix": daughter_num_pix,
                #"daughter_sum_pix": daughter_sum_pix,
                        "pred_primary_num_pix": primary_ionization.sum(), #primary_ionization.sum(),
                "pred_primary_sum_pix": p.depositions[primary_ionization].sum(),
                "true_num_pix": -1,
                "true_sum_pix": -1,
                "true_particle_energy_deposit": -1,
                'true_semantic_type': -1,
                'true_particle_pdg': -1,
                'true_particle_parent_pdg': -1,
                "true_noghost_num_pix": -1,
                "true_noghost_sum_pix": -1,
                "true_noghost_MeV_sum_pix": -1,
                "true_particle_energy_init": -1,
                "true_particle_sed": -1,
                #'volume': tp.volume
                }
            #print("cp1")   
            #print("Heuristic primary ", update_dict['pred_num_pix'], update_dict['pred_primary_num_pix'])

            if not is_data:
                matched=False
                true_ids_r = []
                m = None
                for mp in matched_r2t:
                    true_idx = 1
                    pred_idx = 0
                    if mp[pred_idx] is None or mp[pred_idx].id != p.id or mp[pred_idx].volume_id != p.volume_id: continue
                    if mp[true_idx] is None: continue
                    if not mp[true_idx].is_contained: continue
                    if mp[true_idx].volume_id != p.volume_id: continue
                    if mp[true_idx].semantic_type != MICHL_SHP: continue
                    m = mp[true_idx] 
                    true_ids_r.append([p.id, m.id])
                    true_ids_t = []
                for mp2 in matched_t2r:
                    true_idx = 0
                    pred_idx = 1
                    if mp2[pred_idx] is None or mp2[pred_idx].id != p.id or mp2[pred_idx].volume_id != p.volume_id: continue
                    if mp2[true_idx] is None: continue
                    if not mp2[true_idx].is_contained: continue
                    if mp2[true_idx].volume_id != p.volume_id: continue
                    if mp2[true_idx].semantic_type != MICHL_SHP: continue 
                    true_ids_t.append([p.id, m.id])
                    #print("cp2")    
                if len(true_ids_r)==1 and len(true_ids_t)==1 and true_ids_r==true_ids_t:
                    #print("cp3")
                    #print(len(true_ids_r), len(true_ids_t), true_ids_r==true_ids_t, true_ids_r, m.truth_id)
                    pts = -1 #sed[sed[:,6]==m.truth_id]
                    #print("sed len: ", len(pts))
                    sed_sum = -1 #pts[:,4].sum()

                    update_dict.update({
                        'matched': True,
                        "true_num_pix": m.size,
                        "true_sum_pix": m.depositions.sum(),
                        "true_particle_energy_deposit": m.energy_deposit,
                        'true_semantic_type': m.semantic_type,
                        'true_particle_pdg': m.pdg_code,
                        'true_particle_parent_pdg': m.parent_pdg_code,
                        "true_noghost_num_pix": m.truth_points.shape[0],
                        "true_noghost_sum_pix": m.truth_depositions.sum(),
                        "true_noghost_MeV_sum_pix": m.truth_depositions_MeV.sum(),
                        "true_particle_energy_init": m.energy_init,
                        "true_particle_sed": sed_sum,
                    })
                    #break

            michels.append(OrderedDict(update_dict)) 

    return [michels, true_michels]

