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

# Maintenant, vous pouvez importer le module `MMPoseInferencer`
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

def perform_inference(video_frames, new_nb_frames, nom, file, download_checkpoints):

    nb_frames = len(video_frames)
    frame_interval = nb_frames // new_nb_frames

    results = []

    for i in range(new_nb_frames):

        # Rediriger la sortie vers un objet "devnull" (un objet qui ne fait rien)
        sys.stdout = open('/dev/null', 'w')  # Sur les systèmes UNIX
        # Appel de la fonction qui génère le message d'erreur
        frame = video_frames[i * frame_interval]
        inferencer = MMPoseInferencer(pose2d=file, pose2d_weights=download_checkpoints, device='cpu')
        result_generator = inferencer(frame, return_vis=True, out_dir='mmpose/vis_results/')
        # Restaurer la sortie standard
        sys.stdout = sys.__stdout__

        result = next(result_generator)
        print(i)
        results.append(result)

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

        return normalized_keypoints
    
    normalized_keypoints = center_and_scale(keypoints, w, h)

    # Calcul des positions relatives des keypoints par rapport à un point de référence
    keypoints_rel = normalized_keypoints.copy()
    keypoints_rel[:,10,:] = keypoints_rel[:,10,:] - (normalized_keypoints[:,12,:] + (normalized_keypoints[:,6,:] - normalized_keypoints[:,12,:]) / 2)
    keypoints_rel[:,16,:] = keypoints_rel[:,16,:] - (normalized_keypoints[:,12,:] + (normalized_keypoints[:,6,:] - normalized_keypoints[:,12,:]) / 2)

    return keypoints, normalized_keypoints, keypoints_rel

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
                        duration=400,  # Durée de chaque image en millisecondes
                        loop=0)  # Nombre de fois que le GIF doit être lu (0 pour une boucle infinie)

def gif_keypoints(results, keypoints, nom):

    (h,w,z) = results[0]['visualization'][0].shape

    keypoints_connections = [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),  # tête (orange)
        (3, 5), (5, 6), (4, 6), (5, 11), (6, 12), (11, 12), # corps (bleu)
        (6, 8), (5, 7), (8, 10), (7, 9), # Bras (vert)
        (12, 14), (11, 13), (14, 16), (13, 15) # Jambes (vert)
    ]

    point_between_ankle = keypoints[:,11,:] + (keypoints[:,12,:]- keypoints[:,11,:])/2

    keypoints_rel2 = keypoints.copy()

    # Coordonnées x et y
    x = keypoints[:,:,0]
    y = keypoints[:,:,1]

    #ajout du point entre les deux hanches 
    point_between_ankle = keypoints[:,11,:] + (keypoints[:,12,:]- keypoints[:,11,:])/2

    keypoints_rel2[:,13,:] = keypoints_rel2[:,13,:] - point_between_ankle
    keypoints_rel2[:,14,:] = keypoints_rel2[:,14,:] - point_between_ankle

    x_left_knee = keypoints_rel2[:,13,0] 
    x_right_knee = keypoints_rel2[:,14,0]

    y_left_knee = keypoints_rel2[:,13,1] 
    y_right_knee = keypoints_rel2[:,14,1]

    angle_genoux = calculer_angle_relatif(x_right_knee, y_right_knee, x_left_knee, y_left_knee)

    keypoints_rel3 = keypoints.copy()

    # Coordonnées x et y
    x = keypoints[:,:,0]
    y = keypoints[:,:,1]

    keypoints_rel3[:,8,:] = keypoints_rel3[:,8,:] - keypoints_rel3[:,6,:]
    keypoints_rel3[:,12,:] = keypoints_rel3[:,12,:] - keypoints_rel3[:,6,:]

    x_right_arm = keypoints_rel3[:,8,0] 
    x_right_ankle = keypoints_rel3[:,12,0]

    y_right_arm = keypoints_rel3[:,8,1] 
    y_right_ankle = keypoints_rel3[:,12,1] 
        
    angle_bras = calculer_angle_relatif(x_right_arm, y_right_arm, x_right_ankle, y_right_ankle)

    n = len(results)
    m = n//2

    liste_images =[]

    # Boucle à travers les images et les keypoints
    for i in range(m):
            fig, ax = plt.subplots(figsize = (5,6))  # Définir la taille de la figure

            keypoint = keypoints[2*i,:,:]
            bbox = results[2*i]['predictions'][0][0]['bbox'][0]
            (h, w, z) = results[2*i]['visualization'][0].shape

            ax.scatter(keypoint[:, 0], keypoint[:, 1], s=10)

            x = np.array([bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]])
            y = np.array([bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]])
            y = h - y

            ax.scatter(x, y, s=1)
            ax.plot(x, y, '-')

            ax.quiver(point_between_ankle[2*i,0], point_between_ankle[2*i,1], x_right_knee[2*i], y_right_knee[2*i], angles='xy', scale_units='xy', scale=1, color='r', label='Right Leg')
            ax.quiver(point_between_ankle[2*i,0], point_between_ankle[2*i,1], x_left_knee[2*i], y_left_knee[2*i], angles='xy', scale_units='xy', scale=1, color='r', label='Left Leg')
            ax.text(point_between_ankle[2*i,0] + 10, point_between_ankle[2*i,1] + 10, f"Angle 1: {angle_genoux[2*i]:.2f} deg", color='r', fontsize=8)

            ax.quiver(keypoints[2*i,6,0], keypoints[2*i,6,1], x_right_arm[2*i], y_right_arm[2*i], angles='xy', scale_units='xy', scale=1, color='black', label='Right Arm')
            ax.quiver(keypoints[2*i,6,0], keypoints[2*i,6,1], x_right_ankle[2*i], y_right_ankle[2*i], angles='xy', scale_units='xy', scale=1, color='black', label='Right Ankle')
            ax.text(keypoints[2*i,6,0] - 200, keypoints[2*i,6,1] , f"Angle 2: {angle_bras[2*i]:.2f} deg", color='black', fontsize=8)

            for connection in keypoints_connections:
                x_start = keypoint[connection[0], 0]
                y_start = keypoint[connection[0], 1]
                x_end = keypoint[connection[1], 0]
                y_end = keypoint[connection[1], 1]
                if connection[0] in [0, 1, 2, 3, 4]:
                    ax.plot([x_start, x_end], [y_start, y_end], color='orange', linestyle='-')
                elif connection[0] in [5, 6, 11, 12]:
                    ax.plot([x_start, x_end], [y_start, y_end], color='blue', linestyle='-')
                else:
                    ax.plot([x_start, x_end], [y_start, y_end], color='green', linestyle='-')

            # Définir les limites des axes x et y pour commencer à 0
            ax.set_xlim(left=0, right=w)
            ax.set_ylim(bottom=0, top=h)

            # Définir le rapport d'aspect à 1
            ax.set_aspect('equal', adjustable='box')

            # Vérifier l'existence du répertoire "_visualisations" et le créer au besoin
            if not os.path.exists("_visualisations"):
                os.mkdir("_visualisations")
            # Vérifier l'existence du sous-répertoire correspondant
            if not os.path.exists(f"_visualisations//{nom}"):
                os.mkdir(f"_visualisations/{nom}")

            from PIL import Image

            # Sauvegarder la figure dans un fichier image
            fig.savefig(f"_visualisations/{nom}/affichage_{i}.png")
            img = Image.open(f"_visualisations/{nom}/affichage_{i}.png")
            liste_images.append(img)
    
    # Vérifier l'existence du répertoire "_gif" et le créer au besoin
    if not os.path.exists("_gif"):
        os.mkdir("_gif")

    # Vérifier l'existence du sous-répertoire correspondant au nom de la vidéo et le créer au besoin
    if not os.path.exists(f"_gif/{nom}"):
        os.mkdir(f"_gif/{nom}")

    # Enregistrer les images sous forme de GIF
    liste_images[0].save(f"_gif/{nom}/keypoints.gif",
                        save_all=True,
                        append_images=liste_images[1:],
                        duration=400,  # Durée de chaque image en millisecondes
                        loop=0)  # Nombre de fois que le GIF doit être lu (0 pour une boucle infinie)
    
def visualisation_premiere_et_derniere_frame(results,nom):

    # FRAME 1
    (h, w, z) = results[0]['visualization'][0].shape

    bbox = results[0]['predictions'][0][0]['bbox'][0]

    x = np.array([bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]])
    y = np.array([bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]])
    y = h - y

    keypoints = np.array(results[0]['predictions'][0][0]["keypoints"])

    # Créer un objet Figure et un objet Axes avec 1 ligne et 2 colonnes
    fig, ax = plt.subplots(1, 2, figsize=(10,5)) # Ajout de la taille de la figure

    # Tracer les points pour la première image
    ax[0].scatter(x, y, s=1)
    ax[0].plot(x, y, '-')

    ax[0].scatter(keypoints[:, 0], h-keypoints[:, 1], s=10)

    # Tableau de correspondances pour relier les keypoints consécutifs
    keypoints_connections = [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),  # tête (orange)
        (3, 5), (5, 6), (4, 6), (5, 11), (6, 12), (11, 12), # corps (bleu)
        (6, 8), (5, 7), (8, 10), (7, 9), # Bras (vert)
        (12, 14), (11, 13), (14, 16), (13, 15) # Jambes (vert)
    ]

    # Relier les keypoints consécutifs pour former le squelette humain
    for connection in keypoints_connections:
        x_start = keypoints[connection[0], 0]
        y_start = h - keypoints[connection[0], 1]
        x_end = keypoints[connection[1], 0]
        y_end = h - keypoints[connection[1], 1]
        
        if connection[0] in [0, 1, 2, 3, 4]:
            ax[0].plot([x_start, x_end], [y_start, y_end], color='orange', linestyle='-')
        elif connection[0] in [5, 6, 11, 12]:
            ax[0].plot([x_start, x_end], [y_start, y_end], color='blue', linestyle='-')
        else:
            ax[0].plot([x_start, x_end], [y_start, y_end], color='green', linestyle='-')

    # Définir les limites des axes x et y pour commencer à 0 pour la première image
    ax[0].set_xlim(left=0, right=w)
    ax[0].set_ylim(bottom=0, top=h)

    # Définir le rapport d'aspect à 1 pour la première image
    ax[0].set_aspect('equal', adjustable='box')

    ax[0].set_title("Affichage de l'extraction des keypoints", y=-0.15)

    # Définir les limites des axes x et y pour commencer à 0 pour la deuxième image
    ax[1].set_xlim(left=0, right=w)
    ax[1].set_ylim(bottom=0, top=h)

    # Définir le rapport d'aspect à 1 pour la deuxième image
    ax[1].set_aspect('equal', adjustable='box')

    # Retourner l'image horizontalement
    ax[1].imshow(results[0]['visualization'][0][::-1, :, :]) #image retournée
    ax[1].set_title("Affichage de la prédiction", y=-0.15)

    # Vérifier l'existence du répertoire "_visualisations" et le créer au besoin
    if not os.path.exists("_visualisations"):
        os.mkdir("_visualisations")

    # Vérifier l'existence du sous-répertoire correspondant au nom de la vidéo et le créer au besoin
    if not os.path.exists(f"_visualisations/{nom}"):
        os.mkdir(f"_visualisations/{nom}")

    # Définir un titre 
    fig.suptitle('Visualisation de la première frame de la vidéo', fontsize=14, fontweight='bold')
    fig.savefig('frame1.png')
        # Sauvegarder la figure dans un fichier image
    fig.savefig(f'_visualisations/{nom}/premiere_frame.png')

    # FRAME -1
    bbox = results[-1]['predictions'][0][0]['bbox'][0]

    x = np.array([bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]])
    y = np.array([bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]])
    y = h - y

    keypoints = np.array(results[-1]['predictions'][0][0]["keypoints"])

    # Créer un objet Figure et un objet Axes avec 1 ligne et 2 colonnes
    fig, ax = plt.subplots(1, 2, figsize=(10,5)) # Ajout de la taille de la figure

    # Tracer les points pour la première image
    ax[0].scatter(x, y, s=1)
    ax[0].plot(x, y, '-')

    ax[0].scatter(keypoints[:, 0], h-keypoints[:, 1], s=10)

    # Tableau de correspondances pour relier les keypoints consécutifs
    keypoints_connections = [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),  # tête (orange)
        (3, 5), (5, 6), (4, 6), (5, 11), (6, 12), (11, 12), # corps (bleu)
        (6, 8), (5, 7), (8, 10), (7, 9), # Bras (vert)
        (12, 14), (11, 13), (14, 16), (13, 15) # Jambes (vert)
    ]

    # Relier les keypoints consécutifs pour former le squelette humain
    for connection in keypoints_connections:
        x_start = keypoints[connection[0], 0]
        y_start = h - keypoints[connection[0], 1]
        x_end = keypoints[connection[1], 0]
        y_end = h - keypoints[connection[1], 1]
        
        if connection[0] in [0, 1, 2, 3, 4]:
            ax[0].plot([x_start, x_end], [y_start, y_end], color='orange', linestyle='-')
        elif connection[0] in [5, 6, 11, 12]:
            ax[0].plot([x_start, x_end], [y_start, y_end], color='blue', linestyle='-')
        else:
            ax[0].plot([x_start, x_end], [y_start, y_end], color='green', linestyle='-')

    # Définir les limites des axes x et y pour commencer à 0 pour la première image
    ax[0].set_xlim(left=0, right=w)
    ax[0].set_ylim(bottom=0, top=h)

    # Définir le rapport d'aspect à 1 pour la première image
    ax[0].set_aspect('equal', adjustable='box')

    ax[0].set_title("Affichage de l'extraction des keypoints", y=-0.15)

    # Définir les limites des axes x et y pour commencer à 0 pour la deuxième image
    ax[1].set_xlim(left=0, right=w)
    ax[1].set_ylim(bottom=0, top=h)

    # Définir le rapport d'aspect à 1 pour la deuxième image
    ax[1].set_aspect('equal', adjustable='box')

    # Retourner l'image horizontalement
    ax[1].imshow(results[-1]['visualization'][0][::-1, :, :]) #image retournée
    ax[1].set_title("Affichage de la prédiction", y=-0.15)

    # Définir un titre 
    fig.suptitle('Visualisation de la dernière frame de la vidéo', fontsize=14, fontweight='bold')
    # Sauvegarder la figure dans un fichier image
    fig.savefig('frame-1.png')
    # Sauvegarder la figure dans un fichier image
    fig.savefig(f'_visualisations/{nom}/derniere_frame.png')

    # Charger les images des deux frames
    image_frame1 = plt.imread('frame1.png')
    image_frame2 = plt.imread('frame-1.png')

    # Créer une figure avec deux sous-graphiques
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Afficher la première image dans le premier sous-graphique
    axs[0].imshow(image_frame1)
    axs[0].axis('off')

    # Afficher la deuxième image dans le deuxième sous-graphique
    axs[1].imshow(image_frame2)
    axs[1].axis('off')

    # Définir un titre pour la figure
    fig.suptitle('Visualisation de la première et de la dernière frame de la vidéo', fontsize=14, fontweight='bold')

    # Ajuster les espaces entre les sous-graphiques
    fig.tight_layout()

    # Vérifier l'existence du répertoire "_gif" et le créer au besoin
    if not os.path.exists("_visualisations"):
        os.mkdir("_visualisations")
    # Vérifier l'existence du sous-répertoire correspondant
    if not os.path.exists(f"_visualisations//{nom}"):
        os.mkdir(f"_visualisations/{nom}")

    # Sauvegarder la figure dans un fichier image
    fig.savefig(f'_visualisations/{nom}/premiere_et_derniere_frame.png')

    # Vérifier si le fichier existe avant de le supprimer
    if os.path.exists('frame-1.png'):
        # Supprimer le fichier
        os.remove('frame-1.png')
    else:
        print("Le fichier n'existe pas.")

    # Vérifier si le fichier existe avant de le supprimer
    if os.path.exists('frame1.png'):
        # Supprimer le fichier
        os.remove('frame1.png')
    else:
        print("Le fichier n'existe pas.")

def visualisation_mouvement(results, keypoints, nom):

    (h, w, z) = results[0]['visualization'][0].shape

    keypoints_connections = [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),  # tête (orange)
        (3, 5), (5, 6), (4, 6), (5, 11), (6, 12), (11, 12), # corps (bleu)
        (6, 8), (5, 7), (8, 10), (7, 9), # Bras (vert)
        (12, 14), (11, 13), (14, 16), (13, 15) # Jambes (vert)
    ]

    bbox = results[-1]['predictions'][0][0]['bbox'][0]

    x = np.array([bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]])
    y = np.array([bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]])
    y = h - y

    # Créer un objet Figure et un objet Axes avec 1 ligne et 2 colonnes
    fig, ax = plt.subplots(2, 4, figsize=(20,10)) # Ajout de la taille de la figure

    n = len(results)
    m = n//8

    # Boucle à travers les images et les keypoints
    for i in range(2):
        for j in range(4):
            # Tracer les points
            if i == 0 and j==0:
                keypoints = np.array(results[0]['predictions'][0][0]["keypoints"])
                bbox = results[0]['predictions'][0][0]['bbox'][0]
            elif i == 0 and j == 1:
                keypoints = np.array(results[m]['predictions'][0][0]["keypoints"])
                bbox = results[m]['predictions'][0][0]['bbox'][0]
            elif i == 0 and j == 2:
                keypoints = np.array(results[2*m]['predictions'][0][0]["keypoints"])
                bbox = results[2*m]['predictions'][0][0]['bbox'][0]
            elif i == 0 and j == 3:
                keypoints = np.array(results[3*m]['predictions'][0][0]["keypoints"])
                bbox = results[3*m]['predictions'][0][0]['bbox'][0]
            elif i == 1 and j == 0:
                keypoints = np.array(results[4*m]['predictions'][0][0]["keypoints"])
                bbox = results[4*m]['predictions'][0][0]['bbox'][0]
            elif i == 1 and j == 1:
                keypoints = np.array(results[5*m]['predictions'][0][0]["keypoints"])
                bbox = results[5*m]['predictions'][0][0]['bbox'][0]
            elif i == 1 and j == 2:
                keypoints = np.array(results[6*m]['predictions'][0][0]["keypoints"])
                bbox = results[6*m]['predictions'][0][0]['bbox'][0]
            elif i == 1 and j == 3:
                keypoints = np.array(results[7*m]['predictions'][0][0]["keypoints"])
                bbox = results[7*m]['predictions'][0][0]['bbox'][0]
                
            x = np.array([bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]])
            y = np.array([bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]])
            y = h - y
            ax[i][j].scatter(x, y, s=1)
            ax[i][j].plot(x, y, '-')

            ax[i][j].scatter(keypoints[:, 0], h-keypoints[:, 1], s=10)

            for connection in keypoints_connections:
                x_start = keypoints[connection[0], 0]
                y_start = h - keypoints[connection[0], 1]
                x_end = keypoints[connection[1], 0]
                y_end = h - keypoints[connection[1], 1]
                if connection[0] in [0, 1, 2, 3, 4]:
                    ax[i][j].plot([x_start, x_end], [y_start, y_end], color='orange', linestyle='-')
                elif connection[0] in [5, 6, 11, 12]:
                    ax[i][j].plot([x_start, x_end], [y_start, y_end], color='blue', linestyle='-')
                else:
                    ax[i][j].plot([x_start, x_end], [y_start, y_end], color='green', linestyle='-')

            # Définir les limites des axes x et y pour commencer à 0
            ax[i][j].set_xlim(left=0, right=w)
            ax[i][j].set_ylim(bottom=0, top=h)

            # Définir le rapport d'aspect à 1
            ax[i][j].set_aspect('equal', adjustable='box')

    # Vérifier l'existence du répertoire "_visualisations" et le créer au besoin
    if not os.path.exists("_visualisations"):
        os.mkdir("_visualisations")
    # Vérifier l'existence du sous-répertoire correspondant
    if not os.path.exists(f"_visualisations//{nom}"):
        os.mkdir(f"_visualisations/{nom}")

    # Sauvegarder la figure dans un fichier image
    fig.savefig(f"_visualisations/{nom}/mouvement.png")

def position_keypoints_rel(keypoints_lite_rel, nom):
    fig, ax = plt.subplots()

    # Tracer les points
    ax.scatter(keypoints_lite_rel[:,16,0], keypoints_lite_rel[:,16,1], s=20, marker='.', color='blue', label='Pied droit')
    ax.scatter(keypoints_lite_rel[:,10,0], keypoints_lite_rel[:,10,1], s=20, marker='.', color='red', label='Main droite')

    ax.legend()

    # ax.set_xlim(left=0, right=w)
    # ax.set_ylim(bottom=0, top=h)
    # Ajouter des axes et des étiquettes
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Coordonnée X')
    ax.set_ylabel('Coordonnée Y')
    ax.set_title('Analyse du mouvement (keypoints_rel)')

    # Améliorer la visibilité
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Vérifier l'existence du répertoire "_visualisations" et le créer au besoin
    if not os.path.exists("_visualisations"):
        os.mkdir("_visualisations")
    # Vérifier l'existence du sous-répertoire correspondant
    if not os.path.exists(f"_visualisations//{nom}"):
        os.mkdir(f"_visualisations/{nom}")

    # Sauvegarder la figure dans un fichier image
    fig.savefig(f'_visualisations/{nom}/position_keypoints_rel.png')

def position_keypoints(results, keypoints, nom):
    (h, w, z) = results[0]['visualization'][0].shape

    fig, ax = plt.subplots()

    # Tracer les points
    ax.scatter(keypoints[:,16,0], keypoints[:,16,1], s=20, marker='.', color='blue', label='Pied droit')
    ax.scatter(keypoints[:,10,0], keypoints[:,10,1], s=20, marker='.', color='red', label='Main droite')
    ax.scatter(keypoints[:,12,0]+(keypoints[:,6,0] - keypoints[:,12,0])/2, keypoints[:,12,1]+(keypoints[:,6,1] - keypoints[:,12,1])/2, s=20, marker='.', color='green', label='Point de centrage')

    ax.legend()

    ax.set_xlim(left=0, right=w)
    ax.set_ylim(bottom=0, top=h)
    # Ajouter des axes et des étiquettes
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Coordonnée X')
    ax.set_ylabel('Coordonnée Y')
    ax.set_title('Analyse du mouvement (keypoints)')

    # Améliorer la visibilité
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Vérifier l'existence du répertoire "_visualisations" et le créer au besoin
    if not os.path.exists("_visualisations"):
        os.mkdir("_visualisations")
    # Vérifier l'existence du sous-répertoire correspondant
    if not os.path.exists(f"_visualisations//{nom}"):
        os.mkdir(f"_visualisations/{nom}")

    # Sauvegarder la figure dans un fichier image
    fig.savefig(f'_visualisations/{nom}/position_keypoints.png')

def vecteur_vitesse(course, nom):
    # Calculer le vecteur vitesse
    vel = (course[1:,:,:] - course[:-1,:,:])

    # Créer un objet Figure et un objet Axes
    fig, ax = plt.subplots(figsize=(8, 6))  # Définir la taille de la figure

    # Tracer les points
    ax.scatter(course[:, 16, 0], course[:, 16, 1], s=50, color='blue', label='Points')

    # Tracer les flèches représentant le vecteur vitesse
    ax.quiver(
        course[:-1, 16, 0], 
        course[:-1, 16, 1], 
        vel[:, 16, 0], 
        vel[:, 16, 1], 
        angles='xy', 
        scale_units='xy', 
        scale=2,
        color='red',
        label='Vitesse'
    )

    # Ajouter un titre au graphique
    ax.set_title("Mouvement du pied droit")

    # Ajouter des étiquettes d'axes
    ax.set_xlabel("Position x")
    ax.set_ylabel("Position y")

    # Afficher une légende
    ax.legend()

    # Afficher une grille de fond
    ax.grid(True, linestyle='--', alpha=0.5)

    # Afficher le graphique
    plt.tight_layout()  # Améliorer l'espacement entre les éléments du graphique

    # Vérifier l'existence du répertoire "_visualisations" et le créer au besoin
    if not os.path.exists("_visualisations"):
        os.mkdir("_visualisations")
    # Vérifier l'existence du sous-répertoire correspondant
    if not os.path.exists(f"_visualisations//{nom}"):
        os.mkdir(f"_visualisations/{nom}")

    # Sauvegarder la figure dans un fichier image
    fig.savefig(f'_visualisations/{nom}/vecteur_vitesse.png')

def amplitude_bras_droit(keypoints):
    x_right_hand = keypoints[:,10,0]
    y_right_hand = keypoints[:,10,1]

    amplitude_y_right_hand = np.sqrt(np.sum((min(y_right_hand)- max(y_right_hand)) ** 2))
    amplitude_x_right_hand = np.sqrt(np.sum((min(x_right_hand)- max(x_right_hand)) ** 2))
    return amplitude_x_right_hand, amplitude_y_right_hand

def amplitude_pied_droit(keypoints):
    x_right_foot = keypoints[:,16,0]
    y_right_foot = keypoints[:,16,1]

    amplitude_y_right_foot = np.sqrt(np.sum((min(y_right_foot)- max(y_right_foot)) ** 2))
    amplitude_x_right_foot = np.sqrt(np.sum((min(x_right_foot)- max(x_right_foot)) ** 2))
    return amplitude_x_right_foot, amplitude_y_right_foot

def calculer_angle_relatif(x1, y1, x2, y2):
    # Calculer l'angle en radians
    angle_radians = np.arctan2(y2, x2) - np.arctan2(y1, x1)
    # Ajouter 2*pi pour obtenir un angle dans l'intervalle [-pi, pi]
    angle_radians = (angle_radians + 2*np.pi) % (2*np.pi)
    angle_degrees = angle_radians * 180 / np.pi
    angle_degrees = (angle_degrees + 180) % 360 - 180  # Ajuster la plage de -180 à 180 degrés
    return angle_degrees

def calculer_angle_genoux(keypoints, nom):
    keypoints_rel2 = keypoints.copy()

    # Coordonnées x et y
    x = keypoints[:,:,0]
    y = keypoints[:,:,1]

    #ajout du point entre les deux hanches 
    point_between_ankle = keypoints[:,11,:] + (keypoints[:,12,:]- keypoints[:,11,:])/2

    keypoints_rel2[:,13,:] = keypoints_rel2[:,13,:] - point_between_ankle
    keypoints_rel2[:,14,:] = keypoints_rel2[:,14,:] - point_between_ankle

    x_left_knee = keypoints_rel2[:,13,0] 
    x_right_knee = keypoints_rel2[:,14,0]

    y_left_knee = keypoints_rel2[:,13,1] 
    y_right_knee = keypoints_rel2[:,14,1]

    x1 = x_right_knee 
    y1 = y_right_knee 
    x2 = x_left_knee
    y2 = y_left_knee

    angles = calculer_angle_relatif(x1, y1, x2, y2)

    # Tracer la courbe des angles des genoux
    temps = np.arange(len(angles))  # numéro de l'image
    # Vérifier l'existence du répertoire "_visualisations" et le créer au besoin
    if not os.path.exists("_visualisations"):
        os.mkdir("_visualisations")
    # Vérifier l'existence du sous-répertoire correspondant
    if not os.path.exists(f"_visualisations//{nom}"):
        os.mkdir(f"_visualisations/{nom}")

    fig, ax = plt.subplots()
    ax.plot(temps, angles, "b:o")
    ax.set_xlabel('Numéro de la frame')
    ax.set_ylabel('Angle des genoux (°)')
    ax.set_title('Évolution de l\'angle des genoux au fil du mouvement')

    # Afficher une grille de fond
    ax.grid(True, linestyle='--', alpha=0.5)

    # Afficher le graphique
    plt.tight_layout()  # Améliorer l'espacement entre les éléments du graphique

    fig.savefig(f'_visualisations/{nom}/angle_genoux.png')

    return angles

def calculer_angle_bras_droit(keypoints, nom):
    keypoints_rel3 = keypoints.copy()

    # Coordonnées x et y
    x = keypoints[:,:,0]
    y = keypoints[:,:,1]

    keypoints_rel3[:,8,:] = keypoints_rel3[:,8,:] - keypoints_rel3[:,6,:]
    keypoints_rel3[:,12,:] = keypoints_rel3[:,12,:] - keypoints_rel3[:,6,:]

    x_right_arm = keypoints_rel3[:,8,0] 
    x_right_ankle = keypoints_rel3[:,12,0]

    y_right_arm = keypoints_rel3[:,8,1] 
    y_right_ankle = keypoints_rel3[:,12,1]

    angles = calculer_angle_relatif(x_right_arm, y_right_arm, x_right_ankle, y_right_ankle)
    temps = np.arange(len(angles))  # numéro de l'image

    # Vérifier l'existence du répertoire "_visualisations" et le créer au besoin
    if not os.path.exists("_visualisations"):
        os.mkdir("_visualisations")
    # Vérifier l'existence du sous-répertoire correspondant
    if not os.path.exists(f"_visualisations//{nom}"):
        os.mkdir(f"_visualisations/{nom}")

    fig, ax = plt.subplots()
    ax.plot(temps, angles, "r:o")
    ax.set_xlabel('Numéro de la frame')
    ax.set_ylabel('Angle du bras (°)')
    ax.set_title("Évolution de l\'angle entre l'avant bras droit et la hanche")

    # Afficher une grille de fond
    ax.grid(True, linestyle='--', alpha=0.5)

    # Afficher le graphique
    plt.tight_layout()  # Améliorer l'espacement entre les éléments du graphique
    
    fig.savefig(f'_visualisations/{nom}/angle_bras_droit.png')
    
    return angles

def human_pose_estimation(video_path):

    video_frames = load_video_frames(video_path)

    def get_video_name(video_path):
        file_name = os.path.basename(video_path)  # Obtient le nom du fichier avec l'extension
        video_name = os.path.splitext(file_name)[0]  # Supprime l'extension du nom du fichier
        return video_name
    nom = get_video_name(video_path)
    
    results  = perform_inference(video_frames, 16 , nom, file = 'mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py', download_checkpoints = 'rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth')

    ## Sans inférence
    # with open(f'_results/{nom}/results1_video', 'rb') as f1:
    #     results1 = pickle.load(f1)
    # with open(f'_results/{nom}/results2_video', 'rb') as f1:
    #     results2 = pickle.load(f1)
    # with open(f'_results/{nom}/results3_video', 'rb') as f1:
    #     results3 = pickle.load(f1)
    # with open(f'_results/{nom}/results4_video', 'rb') as f1:
    #     results4 = pickle.load(f1) 
    # results = results1+results2+results3+results4
    
    keypoints, normalized_keypoints, keypoints_rel = process_video_results(results)

    amplitude_x_right_hand, amplitude_y_right_hand = amplitude_bras_droit(keypoints_rel)
    amplitude_x_right_foot, amplitude_y_right_foot = amplitude_pied_droit(keypoints_rel)

    angles_genoux = calculer_angle_genoux(keypoints, nom)
    angles_bras_droit = calculer_angle_bras_droit(keypoints, nom)

    save_images_as_gif(results, nom)
    gif_keypoints(results, keypoints, nom)
    visualisation_premiere_et_derniere_frame(results, nom)
    visualisation_mouvement(results, keypoints, nom)
    position_keypoints(results, keypoints, nom)
    position_keypoints_rel(keypoints_rel, nom)
    vecteur_vitesse(keypoints, nom)

    outputs = {'amplitude_x_right_hand' : amplitude_x_right_hand,
               'amplitude_y_right_hand' : amplitude_y_right_hand,
               'amplitude_x_right_foot' : amplitude_x_right_foot,
               'amplitude_y_right_foot' : amplitude_y_right_foot,
                'angle_max_genoux' : np.max(angles_genoux),
                'angle_min_genoux' : np.min(angles_genoux),
                'angle_max_bras_droit' : np.max(angles_bras_droit),
                'angle_min_bras_droit' : np.min(angles_bras_droit)}
    return outputs 

def main():

    chemin_du_script = os.path.abspath(__file__)
    repertoire_du_projet = os.path.dirname(os.path.dirname(os.path.dirname(chemin_du_script)))
    print(f"La fonction hpe est exectuée à l'emplacement : {repertoire_du_projet}")

    video_path = f'{repertoire_du_projet}/_videos/Bolt_cycle.mp4'
    if os.path.exists(video_path):
        print(f'Le chemin de la vidéo est : {video_path}')
    

    outputs = human_pose_estimation(video_path)

if __name__ == '__main__':
    main()