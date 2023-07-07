
from scripts.hpe import human_pose_estimation
from scripts.comparaison import comparaison

import os
import shutil
import imageio
import matplotlib 
matplotlib.use('Agg')  # Configuration de Matplotlib en mode non interactif
from PIL import Image, ImageSequence
from moviepy.editor import VideoFileClip

from flask import Flask, render_template, request
app = Flask(__name__, static_folder='static')

@app.route('/')
def home():
    return render_template('index.html')

def get_video_name(video_path):
    file_name = os.path.basename(video_path)  # Obtient le nom du fichier avec l'extension
    video_name = os.path.splitext(file_name)[0]  # Supprime l'extension du nom du fichier
    return video_name

def get_video_duration(video_path):
    # Utilisation de moviepy pour extraire la durée de la vidéo
    clip = VideoFileClip(video_path)
    duration = clip.duration
    clip.close()
    return duration

@app.route('/process', methods=['POST'])
def process():
    video = request.files['video1']

    # Sauvegarde du fichier vidéo
    video.save(video.filename)
    print(video.filename)
    video_path = os.path.abspath(video.filename)
    print(video_path)

    # Vérification de la durée de la vidéo
    video_duration = get_video_duration(video_path)
    max_duration = 10  # Durée maximale en secondes (10 minutes)
    if video_duration > max_duration:
        # Suppression du fichier vidéo temporaire
        os.remove(video_path)
        return "La durée de la vidéo est trop longue. Veuillez sélectionner une vidéo de moins de 10 secondes. "
    
    video_2 = request.files['video2']

    if video_2.filename=='':

        # Appel de la fonction d'analyse en utilisant les variables nom et video_path
        outputs = human_pose_estimation(video_path)

        video_name = get_video_name(video_path)

        # Sauvegarde des resultats
        static_dir = "_application_lite/static"
        print(os.path.abspath(static_dir))

        destination_dir = os.path.join(static_dir, video_name)
        destination_path = os.path.join(destination_dir, "video.mp4")
        os.makedirs(destination_dir, exist_ok=True)
        shutil.copyfile(video_path, destination_path)

        # Suppression du fichier vidéo après l'analyse (facultatif)
        os.remove(video_path)

        # Chemins vers les fichiers GIF
        chemin_gif1 = f"_gif/{video_name}/animation.gif"
        chemin_gif2 = f"_gif/{video_name}/keypoints.gif"

        # Ouvrir les GIF
        gif1 = Image.open(chemin_gif1)
        gif2 = Image.open(chemin_gif2)
        # Calculer la taille moyenne
        largeur_moyenne = (gif1.width + gif2.width) // 2
        hauteur_moyenne = (gif1.height + gif2.height) // 2
        # Redimensionner les images individuelles dans le premier GIF
        frames_gif1_redimensionne = []
        for frame in ImageSequence.Iterator(gif1):
            frame_redimensionne = frame.resize((largeur_moyenne, hauteur_moyenne))
            frames_gif1_redimensionne.append(frame_redimensionne)
        # Redimensionner les images individuelles dans le deuxième GIF
        frames_gif2_redimensionne = []
        for frame in ImageSequence.Iterator(gif2):
            frame_redimensionne = frame.resize((largeur_moyenne, hauteur_moyenne))
            frames_gif2_redimensionne.append(frame_redimensionne)
        # Enregistrer le premier GIF redimensionné
        chemin_gif1_redimensionne = chemin_gif1
        frames_gif1_redimensionne[0].save(chemin_gif1_redimensionne, save_all=True, append_images=frames_gif1_redimensionne[1:], loop=0)
        # Enregistrer le deuxième GIF redimensionné
        chemin_gif2_redimensionne = chemin_gif2
        frames_gif2_redimensionne[0].save(chemin_gif2_redimensionne, save_all=True, append_images=frames_gif2_redimensionne[1:], loop=0)


        # Gif 1
        gif_path_absolute = f"_gif/{video_name}/animation.gif"
        destination_dir = os.path.join(static_dir, video_name)
        destination_path = os.path.join(destination_dir, "animation.gif")
        os.makedirs(destination_dir, exist_ok=True)
        shutil.copyfile(gif_path_absolute, destination_path)

        # Gif 2
        gif_path_absolute = f"_gif/{video_name}/keypoints.gif"
        destination_dir = os.path.join(static_dir, video_name)
        destination_path = os.path.join(destination_dir, "keypoints.gif")
        os.makedirs(destination_dir, exist_ok=True)
        shutil.copyfile(gif_path_absolute, destination_path)

        # Visualisation1
        visualisation1_path_absolute = f"_visualisations/{video_name}/premiere_frame.png"
        destination_path = os.path.join(destination_dir, "premiere_frame.png")
        shutil.copyfile(visualisation1_path_absolute, destination_path)

        # Visualisation2
        visualisation2_path_absolute = f"_visualisations/{video_name}/derniere_frame.png"
        destination_path = os.path.join(destination_dir, "derniere_frame.png")
        shutil.copyfile(visualisation2_path_absolute, destination_path)

        # Visualisation3
        visualisation3_path_absolute = f"_visualisations/{video_name}/mouvement.png"
        destination_path = os.path.join(destination_dir, "mouvement.png")
        shutil.copyfile(visualisation3_path_absolute, destination_path)

        # Visualisation4
        visualisation4_path_absolute = f"_visualisations/{video_name}/position_keypoints.png"
        destination_path = os.path.join(destination_dir, "position_keypoints.png")
        shutil.copyfile(visualisation4_path_absolute, destination_path)

        # Visualisation5
        visualisation5_path_absolute = f"_visualisations/{video_name}/position_keypoints_rel.png"
        destination_path = os.path.join(destination_dir, "position_keypoints_rel.png")
        shutil.copyfile(visualisation5_path_absolute, destination_path)

        # Visualisation6
        visualisation6_path_absolute = f"_visualisations/{video_name}/vecteur_vitesse.png"
        destination_path = os.path.join(destination_dir, "vecteur_vitesse.png")
        shutil.copyfile(visualisation6_path_absolute, destination_path)

        # Visualisation6
        visualisation6_path_absolute = f"_visualisations/{video_name}/angle_bras_droit.png"
        destination_path = os.path.join(destination_dir, "angle_bras_droit.png")
        shutil.copyfile(visualisation6_path_absolute, destination_path)

        # Visualisation7
        visualisation6_path_absolute = f"_visualisations/{video_name}/angle_genoux.png"
        destination_path = os.path.join(destination_dir, "angle_genoux.png")
        shutil.copyfile(visualisation6_path_absolute, destination_path)

        return render_template('results_1_video.html',
                                video_name=video_name, 
                                gif_path_1=f'{video_name}/animation.gif',
                                gif_path_2=f'{video_name}/keypoints.gif',
                                video_path = f'{video_name}/video.mp4',
                                amplitude_x_right_hand = outputs['amplitude_x_right_hand'],
                                amplitude_y_right_hand = outputs['amplitude_y_right_hand'],
                                amplitude_x_right_foot = outputs['amplitude_x_right_foot'],
                                amplitude_y_right_foot = outputs['amplitude_y_right_foot'],
                                angle_max_genoux = outputs['angle_max_genoux'],
                                angle_min_genoux = outputs['angle_min_genoux'],
                                angle_max_bras_droit = outputs['angle_max_bras_droit'],
                                angle_min_bras_droit = outputs['angle_min_bras_droit'],
                                visualisation1_path=f"{video_name}/premiere_frame.png",
                                visualisation2_path=f"{video_name}/derniere_frame.png",
                                visualisation3_path=f"{video_name}/mouvement.png",
                                visualisation4_path=f"{video_name}/position_keypoints.png",
                                visualisation5_path=f"{video_name}/position_keypoints_rel.png",
                                visualisation6_path=f"{video_name}/vecteur_vitesse.png",
                                visualisation7_path=f"{video_name}/angle_bras_droit.png",
                                visualisation8_path=f"{video_name}/angle_genoux.png")
    else:
        # Sauvegarde du fichier vidéo
        video_2.save(video_2.filename)
        print(video_2.filename)
        video_path_2= os.path.abspath(video_2.filename)
        print(video_path_2)

        # Vérification de la durée de la vidéo
        video_duration = get_video_duration(video_path_2)
        max_duration = 10  # Durée maximale en secondes (10 minutes)
        if video_duration > max_duration:
            # Suppression du fichier vidéo temporaire
            os.remove(video_path_2)
            return "La durée de la vidéo est trop longue. Veuillez sélectionner une vidéo de moins de 10 secondes. "
        
        outputs = comparaison(video_path,video_path_2)

        # Suppression du fichier vidéo après l'analyse (facultatif)
        os.remove(video_path)
        if os.path.exists(video_path_2):
            os.remove(video_path_2)
        # Sauvegarde des resultats
        static_dir = "_application_lite/static"

        video_name = get_video_name(video_path)
        video_name_2 = get_video_name(video_path_2)

        destination_dir = os.path.join(static_dir, f'comparaison_{video_name}_{video_name_2}')

        # Chemins vers les fichiers GIF
        chemin_gif1 = f"_gif/{video_name}/animation.gif"
        chemin_gif2 = f"_gif/{video_name_2}/animation.gif"
        # Ouvrir les GIF
        gif1 = Image.open(chemin_gif1)
        gif2 = Image.open(chemin_gif2)
        # Calculer la taille moyenne
        largeur_moyenne = (gif1.width + gif2.width) // 2
        hauteur_moyenne = (gif1.height + gif2.height) // 2
        # Redimensionner les images individuelles dans le premier GIF
        frames_gif1_redimensionne = []
        for frame in ImageSequence.Iterator(gif1):
            frame_redimensionne = frame.resize((largeur_moyenne, hauteur_moyenne))
            frames_gif1_redimensionne.append(frame_redimensionne)
        # Redimensionner les images individuelles dans le deuxième GIF
        frames_gif2_redimensionne = []
        for frame in ImageSequence.Iterator(gif2):
            frame_redimensionne = frame.resize((largeur_moyenne, hauteur_moyenne))
            frames_gif2_redimensionne.append(frame_redimensionne)
        # Enregistrer le premier GIF redimensionné
        chemin_gif1_redimensionne = chemin_gif1
        frames_gif1_redimensionne[0].save(chemin_gif1_redimensionne, save_all=True, append_images=frames_gif1_redimensionne[1:], loop=0)
        # Enregistrer le deuxième GIF redimensionné
        chemin_gif2_redimensionne = chemin_gif2
        frames_gif2_redimensionne[0].save(chemin_gif2_redimensionne, save_all=True, append_images=frames_gif2_redimensionne[1:], loop=0)

        # Gif 1
        gif_path_absolute = f"_gif/{video_name}/animation.gif"
        destination_path = os.path.join(destination_dir, f"animation_{video_name}.gif")
        os.makedirs(destination_dir, exist_ok=True)
        shutil.copyfile(gif_path_absolute, destination_path)

        # Gif 2
        gif_path_absolute = f"_gif/{video_name_2}/animation.gif"
        destination_path = os.path.join(destination_dir, f"animation_{video_name_2}.gif")
        os.makedirs(destination_dir, exist_ok=True)
        shutil.copyfile(gif_path_absolute, destination_path)

        # Visualisation1
        visualisation1_path_absolute = f"_visualisations/comparaison_{video_name}_{video_name_2}/position_keypoints_rel.png"
        destination_path = os.path.join(destination_dir, "position_keypoints_rel.png")
        os.makedirs(destination_dir, exist_ok=True)
        shutil.copyfile(visualisation1_path_absolute, destination_path)

        # Visualisation2
        visualisation2_path_absolute = f"_visualisations/comparaison_{video_name}_{video_name_2}/angle_bras_droit.png"
        destination_path = os.path.join(destination_dir, "angle_bras_droit.png")
        shutil.copyfile(visualisation2_path_absolute, destination_path)

        # Visualisation3
        visualisation3_path_absolute = f"_visualisations/comparaison_{video_name}_{video_name_2}/angle_genoux.png"
        destination_path = os.path.join(destination_dir, "angle_genoux.png")
        shutil.copyfile(visualisation3_path_absolute, destination_path)

        return render_template('results_2_videos.html',
                                gif_path_1=f'comparaison_{video_name}_{video_name_2}/animation_{video_name}.gif',
                                gif_path_2=f'comparaison_{video_name}_{video_name_2}/animation_{video_name_2}.gif',
                                amplitude_x_right_hand_1 = outputs['amplitude_x_right_hand_1'],
                                amplitude_y_right_hand_1 = outputs['amplitude_y_right_hand_1'],
                                amplitude_x_right_foot_1 = outputs['amplitude_x_right_foot_1'],
                                amplitude_y_right_foot_1 = outputs['amplitude_y_right_foot_1'],

                                amplitude_x_right_hand_2 = outputs['amplitude_x_right_hand_2'],
                                amplitude_y_right_hand_2 = outputs['amplitude_y_right_hand_2'],
                                amplitude_x_right_foot_2 = outputs['amplitude_x_right_foot_2'],
                                amplitude_y_right_foot_2 = outputs['amplitude_y_right_foot_2'],

                                ecart_relatif_amplitude_x_right_hand = outputs['ecart_relatif_amplitude_x_right_hand'],
                                ecart_relatif_amplitude_y_right_hand = outputs['ecart_relatif_amplitude_y_right_hand'],
                                ecart_relatif_amplitude_x_right_foot = outputs['ecart_relatif_amplitude_x_right_foot'],
                                ecart_relatif_amplitude_y_right_foot = outputs['ecart_relatif_amplitude_y_right_foot'],
                                ecart_relatif_amplitude_angle_bras_droit = outputs['ecart_relatif_amplitude_angle_bras_droit'],
                                ecart_relatif_amplitude_angle_genoux = outputs['ecart_relatif_amplitude_angle_genoux'],

                                visualisation1_path=f"comparaison_{video_name}_{video_name_2}/position_keypoints_rel.png",
                                visualisation2_path=f"comparaison_{video_name}_{video_name_2}/angle_bras_droit.png",
                                visualisation3_path=f"comparaison_{video_name}_{video_name_2}/angle_genoux.png")

if __name__ == '__main__':
    app.run(debug=True)
