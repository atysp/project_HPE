import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
# from hpe import process_video_results, load_video_frames, perform_inference, save_images_as_gif
##avec main.py
from scripts.hpe import process_video_results, load_video_frames, perform_inference, save_images_as_gif


def comparaison(video_path_1, video_path_2):

    def get_video_name(video_path):
        file_name = os.path.basename(video_path)  # Obtient le nom du fichier avec l'extension
        video_name = os.path.splitext(file_name)[0]  # Supprime l'extension du nom du fichier
        return video_name

    video_frames_1 = load_video_frames(video_path_1)
    nom_1 = get_video_name(video_path_1)
    
    results_1  = perform_inference(video_frames_1, 16 , nom_1, file = 'mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py', download_checkpoints = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth')

    # with open(f'_results/{nom_1}/results1_video', 'rb') as f1:
    #     results1 = pickle.load(f1)
    # with open(f'_results/{nom_1}/results2_video', 'rb') as f1:
    #     results2 = pickle.load(f1)
    # with open(f'_results/{nom_1}/results3_video', 'rb') as f1:
    #     results3 = pickle.load(f1)
    # with open(f'_results/{nom_1}/results4_video', 'rb') as f1:
    #     results4 = pickle.load(f1) 
    # results_1 = results1+results2+results3+results4

    video_frames_2 = load_video_frames(video_path_2)
    nom_2 = get_video_name(video_path_2)

    results_2  = perform_inference(video_frames_2, 16 , nom_2, file = 'mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py', download_checkpoints = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth')

    # with open(f'_results/{nom_2}/results1_video', 'rb') as f1:
    #     results1 = pickle.load(f1)
    # with open(f'_results/{nom_2}/results2_video', 'rb') as f1:
    #     results2 = pickle.load(f1)
    # with open(f'_results/{nom_2}/results3_video', 'rb') as f1:
    #     results3 = pickle.load(f1)
    # with open(f'_results/{nom_2}/results4_video', 'rb') as f1:
    #     results4 = pickle.load(f1) 
    # results_2 = results1+results2+results3+results4

    save_images_as_gif(results_1, nom_1)
    save_images_as_gif(results_2, nom_2)

    keypoints_1, normalized_keypoints_1, keypoints_rel_1 = process_video_results(results_1)
    keypoints_2, normalized_keypoints_2, keypoints_rel_2 = process_video_results(results_2)

    def interpolate_elliptical_trend(coordinates, smoothing_factor=0.02):
        # Convertir les coordonnées en listes séparées de x et y
        x_coords, y_coords = zip(*coordinates)
        
        # Générer un nombre suffisant de points entre le début et la fin de la trajectoire
        num_points = 100
        t = np.linspace(0, 1, num_points)
        
        # Effectuer une interpolation spline avec le facteur de lissage spécifié
        tck, u = interpolate.splprep([x_coords, y_coords], s=smoothing_factor)
        interpolated_coords = interpolate.splev(t, tck)
        return interpolated_coords

    right_hand_rel_1 = interpolate_elliptical_trend(keypoints_rel_1[:,10,:])
    right_hand_rel_2 = interpolate_elliptical_trend(keypoints_rel_2[:,10,:])

    right_foot_rel_1 = interpolate_elliptical_trend(keypoints_rel_1[:,16,:])
    right_foot_rel_2 = interpolate_elliptical_trend(keypoints_rel_2[:,16,:])


    fig, ax = plt.subplots()

    # Tracer les points
    ax.scatter(keypoints_rel_1[:,10,0], keypoints_rel_1[:,10,1], s=20, marker='.', color='red', label=f'Main droite {nom_1}')
    ax.plot(right_hand_rel_1[0], right_hand_rel_1[1], color='red', alpha=0.4)

    ax.scatter(keypoints_rel_1[:,16,0], keypoints_rel_1[:,16,1], s=20, marker='.', color='blue', label=f'Pied droit {nom_1}')
    ax.plot(right_foot_rel_1[0], right_foot_rel_1[1], color='blue', alpha=0.4)

    ax.scatter(keypoints_rel_2[:,10,0], keypoints_rel_2[:,10,1], s=20, marker='.', color='orange', label=f'Main droite {nom_2}')
    ax.plot(right_hand_rel_2[0], right_hand_rel_2[1], color='orange', alpha=0.4)
    ax.scatter(keypoints_rel_2[:,16,0], keypoints_rel_2[:,16,1], s=20, marker='.', color='purple', label=f'Pied droit {nom_2}')
    ax.plot(right_foot_rel_2[0], right_foot_rel_2[1], color='purple', alpha=0.4)

    ax.legend()
    # Ajouter des axes et des étiquettes
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Coordonnée X')
    ax.set_ylabel('Coordonnée Y')
    ax.set_title('Analyse du mouvement (Position des point clefs relative)')

    # Améliorer la visibilité
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Vérifier l'existence du répertoire "_visualisations" et le créer au besoin
    if not os.path.exists("_visualisations"):
        os.mkdir("_visualisations")
        # Vérifier l'existence du sous-répertoire correspondant
    if not os.path.exists(f"_visualisations/comparaison_{nom_1}_{nom_2}"):
        os.mkdir(f"_visualisations/comparaison_{nom_1}_{nom_2}")
    # Vérifier l'existence du sous-répertoire correspondant
    fig.savefig(f'_visualisations/comparaison_{nom_1}_{nom_2}/position_keypoints_rel.png')

    def calculer_angle_relatif(x1, y1, x2, y2):
        # Calculer l'angle en radians
        angle_radians = np.arctan2(y2, x2) - np.arctan2(y1, x1)
        # Ajouter 2*pi pour obtenir un angle dans l'intervalle [-pi, pi]
        angle_radians = (angle_radians + 2*np.pi) % (2*np.pi)
        angle_degrees = angle_radians * 180 / np.pi
        angle_degrees = (angle_degrees + 180) % 360 - 180  # Ajuster la plage de -180 à 180 degrés
        return angle_degrees

    def calculer_angle_genoux(keypoints, nom):

        # Coordonnées x et y
        x = keypoints[:,:,0]
        y = keypoints[:,:,1]

        #ajout du point entre les deux hanches 
        point_between_ankle = keypoints[:,11,:] + (keypoints[:,12,:]- keypoints[:,11,:])/2

        keypoints[:,13,:] = keypoints[:,13,:] - point_between_ankle
        keypoints[:,14,:] = keypoints[:,14,:] - point_between_ankle

        x_left_knee = keypoints[:,13,0] 
        x_right_knee = keypoints[:,14,0]

        y_left_knee = keypoints[:,13,1] 
        y_right_knee = keypoints[:,14,1]

        x1 = x_right_knee 
        y1 = y_right_knee 
        x2 = x_left_knee
        y2 = y_left_knee

        angles = calculer_angle_relatif(x1, y1, x2, y2)
        return angles

    def calculer_angle_bras_droit(keypoints, nom):

        # Coordonnées x et y
        x = keypoints[:,:,0]
        y = keypoints[:,:,1]

        keypoints[:,8,:] = keypoints[:,8,:] - keypoints[:,6,:]
        keypoints[:,12,:] = keypoints[:,12,:] - keypoints[:,6,:]

        x_right_arm = keypoints[:,8,0] 
        x_right_ankle = keypoints[:,12,0]

        y_right_arm = keypoints[:,8,1] 
        y_right_ankle = keypoints[:,12,1]

        angles = calculer_angle_relatif(x_right_arm, y_right_arm, x_right_ankle, y_right_ankle)

        return angles

    angles_genoux_1 = calculer_angle_genoux(keypoints_1, nom_1)
    angles_genoux_2 = calculer_angle_genoux(keypoints_2, nom_2)

    # Vérifier l'existence du répertoire "_visualisations" et le créer au besoin
    if not os.path.exists("_visualisations"):
        os.mkdir("_visualisations")
        # Vérifier l'existence du sous-répertoire correspondant
    if not os.path.exists(f"_visualisations/comparaison_{nom_1}_{nom_2}"):
        os.mkdir(f"_visualisations/comparaison_{nom_1}_{nom_2}")

    temps_1 = np.arange(len(angles_genoux_1))  # numéro de l'image
    temps_2 = np.arange(len(angles_genoux_2)) # numéro de l'image

    if len(angles_genoux_1)>len(angles_genoux_2):
        rapport = len(angles_genoux_1)/len(angles_genoux_2)
        temps_2 = rapport*temps_2
    else:
        rapport = len(angles_genoux_2)/len(angles_genoux_1)
        temps_1 = rapport*temps_1

    fig, ax = plt.subplots()

    ax.plot(temps_1, angles_genoux_1, color = "red", marker = 'o', label=f'{nom_1}')
    ax.plot(temps_2, angles_genoux_2, color = "purple", marker = 'o', label=f'{nom_2}')
    ax.legend()    
    ax.set_xlabel('Numéro de la frame')
    ax.set_ylabel('Angle des genoux (°)')
    ax.set_title("Évolution de l\'angle entre les deux genoux au fil du mouvement")

    # Afficher une grille de fond
    ax.grid(True, linestyle='--', alpha=0.5)

    # Afficher le graphique
    plt.tight_layout()  # Améliorer l'espacement entre les éléments du graphique

    # Vérifier l'existence du sous-répertoire correspondant
    fig.savefig(f'_visualisations/comparaison_{nom_1}_{nom_2}/angle_genoux.png')


    angles_bras_droit_1 = calculer_angle_bras_droit(keypoints_1, nom_1)
    angles_bras_droit_2 = calculer_angle_bras_droit(keypoints_2, nom_2)

    # Vérifier l'existence du répertoire "_visualisations" et le créer au besoin
    if not os.path.exists("_visualisations"):
        os.mkdir("_visualisations")
        # Vérifier l'existence du sous-répertoire correspondant
    if not os.path.exists(f"_visualisations/comparaison_{nom_1}_{nom_2}"):
        os.mkdir(f"_visualisations/comparaison_{nom_1}_{nom_2}")

    temps_1 = np.arange(len(angles_bras_droit_1))  # numéro de l'image
    temps_2 = np.arange(len(angles_bras_droit_2)) # numéro de l'image

    if len(angles_bras_droit_1)>len(angles_bras_droit_2):
        rapport = len(angles_bras_droit_1)/len(angles_bras_droit_2)
        temps_2 = rapport*temps_2
    else:
        rapport = len(angles_bras_droit_2)/len(angles_bras_droit_1)
        temps_1 = rapport*temps_1

    fig, ax = plt.subplots()

    ax.plot(temps_1, angles_bras_droit_1, color = "red", marker = 'o', label=f'{nom_1}')
    ax.plot(temps_2, angles_bras_droit_2, color = "orange", marker = 'o', label=f'{nom_2}')
    ax.legend()
    ax.set_xlabel('Numéro de la frame')
    ax.set_ylabel('Angle du bras (°)')
    ax.set_title("Évolution de l\'angle entre l'avant bras droit et la hanche au fil du mouvement")

    # Afficher une grille de fond
    ax.grid(True, linestyle='--', alpha=0.5)

    # Afficher le graphique
    plt.tight_layout()  # Améliorer l'espacement entre les éléments du graphique

    # Vérifier l'existence du sous-répertoire correspondant
    fig.savefig(f'_visualisations/comparaison_{nom_1}_{nom_2}/angle_bras_droit.png')

    amplitude_x_right_hand_1 = np.sqrt(np.sum((min(keypoints_rel_1[:,10,0])- max(keypoints_rel_1[:,10,0])) ** 2))
    amplitude_y_right_hand_1 = np.sqrt(np.sum((min(keypoints_rel_1[:,10,1])- max(keypoints_rel_1[:,10,1])) ** 2))
    amplitude_x_right_hand_2 = np.sqrt(np.sum((min(keypoints_rel_2[:,10,0])- max(keypoints_rel_2[:,10,0])) ** 2))
    amplitude_y_right_hand_2 = np.sqrt(np.sum((min(keypoints_rel_2[:,10,1])- max(keypoints_rel_2[:,10,1])) ** 2))

    amplitude_x_right_foot_1 = np.sqrt(np.sum((min(keypoints_rel_1[:,16,0])- max(keypoints_rel_1[:,16,0])) ** 2))
    amplitude_y_right_foot_1 = np.sqrt(np.sum((min(keypoints_rel_1[:,16,1])- max(keypoints_rel_1[:,16,1])) ** 2))
    amplitude_x_right_foot_2 = np.sqrt(np.sum((min(keypoints_rel_2[:,16,0])- max(keypoints_rel_2[:,16,0])) ** 2))
    amplitude_y_right_foot_2 = np.sqrt(np.sum((min(keypoints_rel_2[:,16,1])- max(keypoints_rel_2[:,16,1])) ** 2))

    # Score
    difference_amplitude_x_right_hand =  amplitude_y_right_hand_1 - amplitude_y_right_hand_2
    difference_amplitude_y_right_hand =  amplitude_y_right_hand_1 - amplitude_y_right_hand_2

    difference_amplitude_x_right_foot =  amplitude_y_right_foot_1 - amplitude_y_right_foot_2
    difference_amplitude_y_right_foot =  amplitude_y_right_foot_1 - amplitude_y_right_foot_2

    difference_amplitude_angle_bras_droit = (np.max(angles_bras_droit_1)-np.min(angles_bras_droit_1)) - (np.max(angles_bras_droit_2)-np.min(angles_bras_droit_2))
    difference_amplitude_angle_genoux = (np.max(angles_genoux_1)-np.min(angles_genoux_1)) - (np.max(angles_genoux_2)-np.min(angles_genoux_2))

    ecart_relatif_amplitude_x_right_hand = difference_amplitude_x_right_hand / amplitude_x_right_hand_1
    ecart_relatif_amplitude_y_right_hand = difference_amplitude_y_right_hand / amplitude_y_right_hand_1

    ecart_relatif_amplitude_x_right_foot = difference_amplitude_x_right_foot / amplitude_x_right_foot_1
    ecart_relatif_amplitude_y_right_foot = difference_amplitude_y_right_foot / amplitude_y_right_foot_1

    ecart_relatif_amplitude_angle_bras_droit = difference_amplitude_angle_bras_droit / (np.max(angles_bras_droit_1)-np.min(angles_bras_droit_1))
    ecart_relatif_amplitude_angle_genoux = difference_amplitude_angle_genoux / (np.max(angles_genoux_1)-np.min(angles_genoux_1))

    ecart_relatif = ecart_relatif_amplitude_x_right_hand + ecart_relatif_amplitude_y_right_hand + ecart_relatif_amplitude_x_right_foot + ecart_relatif_amplitude_y_right_foot + ecart_relatif_amplitude_angle_bras_droit + ecart_relatif_amplitude_angle_genoux
  

    outputs = {'amplitude_x_right_hand_1' : amplitude_x_right_hand_1,
               'amplitude_y_right_hand_1' : amplitude_y_right_hand_1,
               'amplitude_x_right_foot_1' : amplitude_x_right_foot_1,
               'amplitude_y_right_foot_1' : amplitude_y_right_foot_1,

               'amplitude_x_right_hand_2' : amplitude_x_right_hand_2,
               'amplitude_y_right_hand_2' : amplitude_y_right_hand_2,
               'amplitude_x_right_foot_2' : amplitude_x_right_foot_2,
               'amplitude_y_right_foot_2' : amplitude_y_right_foot_2,

               'ecart_relatif_amplitude_x_right_hand' : ecart_relatif_amplitude_x_right_hand,
               'ecart_relatif_amplitude_y_right_hand' : ecart_relatif_amplitude_y_right_hand,
               'ecart_relatif_amplitude_x_right_foot' : ecart_relatif_amplitude_x_right_foot,
               'ecart_relatif_amplitude_y_right_foot' : ecart_relatif_amplitude_y_right_foot,
               'ecart_relatif_amplitude_angle_bras_droit' : ecart_relatif_amplitude_angle_bras_droit,
               'ecart_relatif_amplitude_angle_genoux' : ecart_relatif_amplitude_angle_genoux,
               'ecart_relatif' : ecart_relatif/6}

    return outputs 

def main():

    chemin_du_script = os.path.abspath(__file__)
    repertoire_du_projet = os.path.dirname(os.path.dirname(os.path.dirname(chemin_du_script)))

    video_path_1 = f'{repertoire_du_projet}/HPE/INPUTS/VIDEOS/Marathon_cycle.mp4'
    video_path_2 =  f'{repertoire_du_projet}/HPE/INPUTS/VIDEOS/Kipkurui_cycle.mp4'

    chemin_du_script = os.path.abspath(__file__)
    repertoire_du_projet = os.path.dirname(os.path.dirname(os.path.dirname(chemin_du_script)))
    print(f"La fonction comparaison est exectuée à l'emplacement : {repertoire_du_projet}")

    video_path_1 = f'{repertoire_du_projet}/_videos/Gardiner_cycle.mp4'
    video_path_2 = f'{repertoire_du_projet}/_videos/Kipkurui_cycle.mp4'
    if os.path.exists(video_path_1):
        print(f'Le chemin des vidéo sont : {video_path_1} et {video_path_2}')
    outputs = comparaison(video_path_1, video_path_2)

    print(outputs['ecart_relatif'])

if __name__ == '__main__':
    main()