import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
import sys

chemin_script = os.path.abspath(__file__)
chemin_script = os.path.dirname(os.path.dirname(os.path.dirname(chemin_script)))
sys.path.append(f"{chemin_script}/mmpose")

from mmpose.apis import MMPoseInferencer

def load_video_frames(video_path):
    video = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        exit()

    while True:
        ret, frame = cap.read()
        frame = np.array(frame)

        if not ret:
            break

        video.append(frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return video

def save_video_frames(video_frames, nom, new_nb_frames = 16):
    nb_frames = len(video_frames)
    frame_interval = nb_frames // new_nb_frames

    if not os.path.exists('_video_frames'):
        os.mkdir("_video_frames")

    if not os.path.exists(f'_video_frames/{nom}'):
        os.mkdir(f"_video_frames/{nom}")

    for i in range(new_nb_frames):
        cv2.imwrite(f"_video_frames/{nom}/frame{i}.png", video_frames[i * frame_interval])
    return new_nb_frames

def perform_and_save_inference(new_nb_frames, nom, file, download_checkpoints):

    if not os.path.exists("_results"):
        os.mkdir("_results")

    if not os.path.exists(f"_results/{nom}"):
        os.mkdir(f"_results/{nom}")

    results1 = []
    partition = new_nb_frames // 4

    for j in range(partition):
        # Rediriger la sortie vers un objet "devnull" (un objet qui ne fait rien)
        sys.stdout = open('/dev/null', 'w')  # Sur les systèmes UNIX
        # Appel de la fonction qui génère le message d'erreur
        frame = f'_video_frames/{nom}/frame{j}.png'
        inferencer = MMPoseInferencer(pose2d=file, pose2d_weights=download_checkpoints, device='cpu')
        result_generator = inferencer(frame, return_vis=True, out_dir='mmpose/vis_results/')
        # Restaurer la sortie standard
        sys.stdout = sys.__stdout__

        result = next(result_generator)
        print(j)
        results1.append(result)

    with open(f'_results/{nom}/results1_video', 'wb') as f1:
        pickle.dump(results1, f1)

    results2 = []
    for j in range(partition, 2 * partition):
        # Rediriger la sortie vers un objet "devnull" (un objet qui ne fait rien)
        sys.stdout = open('/dev/null', 'w')  # Sur les systèmes UNIX
        # Appel de la fonction qui génère le message d'erreur
        frame = f'_video_frames/{nom}/frame{j}.png'
        inferencer = MMPoseInferencer(pose2d=file, pose2d_weights=download_checkpoints, device='cpu')
        result_generator = inferencer(frame, return_vis=True, out_dir='mmpose/vis_results/')
        # Restaurer la sortie standard
        sys.stdout = sys.__stdout__

        result = next(result_generator)
        print(j)
        results2.append(result)

    with open(f'_results/{nom}/results2_video', 'wb') as f1:
        pickle.dump(results2, f1)

    results3 = []
    for j in range(2 * partition, 3 * partition):
        # Rediriger la sortie vers un objet "devnull" (un objet qui ne fait rien)
        sys.stdout = open('/dev/null', 'w')  # Sur les systèmes UNIX
        # Appel de la fonction qui génère le message d'erreur
        frame = f'_video_frames/{nom}/frame{j}.png'
        inferencer = MMPoseInferencer(pose2d=file, pose2d_weights=download_checkpoints, device='cpu')
        result_generator = inferencer(frame, return_vis=True, out_dir='mmpose/vis_results/')
        # Restaurer la sortie standard
        sys.stdout = sys.__stdout__

        result = next(result_generator)
        print(j)
        results3.append(result)

    with open(f'_results/{nom}/results3_video', 'wb') as f1:
        pickle.dump(results3, f1)

    results4 = []
    for j in range(3 * partition, 4 * partition):
        # Rediriger la sortie vers un objet "devnull" (un objet qui ne fait rien)
        sys.stdout = open('/dev/null', 'w')  # Sur les systèmes UNIX
        # Appel de la fonction qui génère le message d'erreur
        frame = f'_video_frames/{nom}/frame{j}.png'
        inferencer = MMPoseInferencer(pose2d=file, pose2d_weights=download_checkpoints, device='cpu')
        result_generator = inferencer(frame, return_vis=True, out_dir='mmpose/vis_results/')
        # Restaurer la sortie standard
        sys.stdout = sys.__stdout__

        result = next(result_generator)
        print(j)
        results4.append(result)

    with open(f'_results/{nom}/results4_video', 'wb') as f1:
        pickle.dump(results4, f1)

def load_results_inferencer(nom):
    results = []
    for i in range(1, 5):
        file_path = f'_results/{nom}/results{i}_video'
        try:
            with open(file_path, 'rb') as f:
                result = pickle.load(f)
                results.extend(result)
        except FileNotFoundError:
            print(f"Le fichier {file_path} n'a pas été trouvé.")
    return results

def process_video_results(results):

    # Récupération des dimensions de la première frame des keypoints visualisés
    (h, w, z) = results[0]['visualization'][0].shape

    # Récupération des keypoints de taille (nb_frame, nb_keypoints, 2) à partir des résultats
    M = np.array([results[i]['predictions'][0][0]["keypoints"] for i in range(len(results))])
    keypoints = M.copy()
    keypoints[:,:,1] = h - keypoints[:,:,1]

    # Calcul de la moyenne des keypoints
    mean = (keypoints[0:-5,:,:] + keypoints[1:-4,:,:] + keypoints[2:-3,:,:] + keypoints[3:-2,:,:] + keypoints[4:-1,:,:] + keypoints[5:,:,:]) / 5
    keypoints_lite = mean[::10,:,:]

    # Normalisation des keypoints
    def center_and_scale(keypoints, w, h):
        # Calcule la longueur caractéristique pour chaque frame
        torso_length = np.linalg.norm(keypoints[:, 6, :] - keypoints[:, 12, :], axis=1)

        # Normalise les tailles des images
        normalized_keypoints = keypoints.copy()
        normalized_keypoints[:, :, 0] = keypoints[:, :, 0] / w * 100
        normalized_keypoints[:, :, 1] = keypoints[:, :, 1] / h * 100

        # Normalise les positions des points clés par rapport au centre de gravité et à la longueur entre l'épaule droit et la hanche droite
        normalized_keypoints = keypoints / torso_length[:, None, None]

        return keypoints
    
    normalized_keypoints = center_and_scale(keypoints_lite, w, h)

    # Calcul des positions relatives des keypoints par rapport à un point de référence
    keypoints_lite_rel = normalized_keypoints.copy()
    keypoints_lite_rel[:,10,:] = keypoints_lite_rel[:,10,:] - (normalized_keypoints[:,12,:] + (normalized_keypoints[:,6,:] - normalized_keypoints[:,12,:]) / 2)
    keypoints_lite_rel[:,16,:] = keypoints_lite_rel[:,16,:] - (normalized_keypoints[:,12,:] + (normalized_keypoints[:,6,:] - normalized_keypoints[:,12,:]) / 2)

    return keypoints, normalized_keypoints, keypoints_lite_rel

def save_images_as_gif(results, nom):
    # Charger les images sous forme de tableaux numpy
    tableaux_images = [results[i*2]['visualization'][0] for i in range(len(results)//2)]

    # Créer une liste pour stocker les objets Image
    liste_images = []
    for tableau_image in tableaux_images:
        # Convertir le tableau numpy en image PIL
        img = Image.fromarray(tableau_image)
        liste_images.append(img)

    # Vérifier l'existence du répertoire "_gif" et le créer au besoin
    if not os.path.exists("_gif"):
        os.mkdir("_gif")

    # Vérifier l'existence du sous-répertoire correspondant au nom de la vidéo et le créer au besoin
    if not os.path.exists(f"_gif/{nom}"):
        os.mkdir(f"_gif/{nom}")

    # Enregistrer les images sous forme de GIF
    liste_images[0].save(f"_gif/{nom}/animation.gif",
                        save_all=True,
                        append_images=liste_images[1:],
                        duration=200,  # Durée de chaque image en millisecondes
                        loop=0)  # Nombre de fois que le GIF doit être lu (0 pour une boucle infinie)

def human_pose_estimation(video_path):

    video_frames = load_video_frames(video_path)

    print(len(video_frames))

    def get_video_name(video_path):
        file_name = os.path.basename(video_path)  # Obtient le nom du fichier avec l'extension
        video_name = os.path.splitext(file_name)[0]  # Supprime l'extension du nom du fichier
        return video_name
    nom = get_video_name(video_path)

    new_nb_frames = save_video_frames(video_frames, nom, new_nb_frames=16)
    
    perform_and_save_inference(new_nb_frames, nom, file = 'mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py', download_checkpoints = 'rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth')

    # results = load_results_inferencer(nom)

    # keypoints, normalized_keypoints, keypoints_lite_rel = process_video_results(results)

    # save_images_as_gif(results, nom)


def main():

    chemin_du_script = os.path.abspath(__file__)
    repertoire_du_projet = os.path.dirname(os.path.dirname(os.path.dirname(chemin_du_script)))
    print(f"La fonction results est exectuée à l'emplacement : {repertoire_du_projet}")

    video_path = f'{repertoire_du_projet}/_videos/rugby3.mp4'
    if os.path.exists(video_path):
        print(f'Le chemin de la vidéo est : {video_path}')

    human_pose_estimation(video_path)

if __name__ == '__main__':
    main()
