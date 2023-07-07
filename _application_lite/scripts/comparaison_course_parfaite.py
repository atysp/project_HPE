import numpy as np

def comparaison_avec_course_parfaite(course, course_parfaite):
        
    # definition des variables
    x_parfaite_right_hand = course_parfaite[:,10,0]
    y_parfaite_right_hand = course_parfaite[:,10,1]
    x_parfaite_right_foot = course_parfaite[:,16,0]
    y_parfaite_right_foot = course_parfaite[:,16,1]
    
    amplitude_y_parfaite = np.sqrt(np.sum((min(y_parfaite_right_hand)- max(y_parfaite_right_hand)) ** 2))
    amplitude_x_parfaite = np.sqrt(np.sum((min(x_parfaite_right_hand)- max(x_parfaite_right_hand)) ** 2))

    x_right_hand = course[:,10,0]
    y_right_hand = course[:,10,1]
    x_right_foot = course[:,16,0]
    y_right_foot = course[:,16,1]
    
    amplitude_y = np.sqrt(np.sum((min(y_right_hand)- max(y_right_hand)) ** 2))
    amplitude_x= np.sqrt(np.sum((min(x_right_hand)- max(x_right_hand)) ** 2))

    def extraire_points(x, y, x_ex, tolérance):
        x_ex_sorted = sorted(x_ex)
        points_ex = []
        if (x_ex_sorted[0] <= x[0]):
            points_ex.append((y[0],y[0]))  
                
        for j in range(0,len(x_ex_sorted)):
            elem = x_ex_sorted[j]
            for i in range(0,len(x)-1):
                if elem >= x[i] and elem < x[i+1]:
                    points_ex.append((y[i],y[i+1]))
        if (x_ex_sorted[len(x_ex_sorted)-1] >= x[len(x)-1]):
            points_ex.append((y[len(x)-1],y[len(x-1)]))    
        return np.array(points_ex)

    y_parfaite_right_hand = extraire_points(x_parfaite_right_hand, y_parfaite_right_hand, x_right_hand, tolérance=0.02)
    y_parfaite_right_foot = extraire_points(x_parfaite_right_foot, y_parfaite_right_foot, x_right_foot, tolérance=0.02)

    def choose(y_parfaite,y):
        y_parfaite_choose = y_parfaite[:,0]
        for i in range(len(y)):
            if  np.abs(y_parfaite[i,1]-y[i]) < np.abs(y_parfaite[i,0]-y[i]):
                y_parfaite_choose[i] = y_parfaite[i,1]
        return y_parfaite_choose

    y_parfaite_right_hand = choose(y_parfaite_right_hand,y_right_hand)
    y_parfaite_right_foot = choose(y_parfaite_right_foot,y_right_foot)

    assert len(y_parfaite_right_hand) == len(x_right_hand)
    assert len(y_parfaite_right_foot) == len(x_right_foot)

    distance_right_hand = np.sqrt(np.sum((y_parfaite_right_hand - y_right_hand) ** 2))
    distance_right_foot = np.sqrt(np.sum((y_parfaite_right_foot - y_right_foot) ** 2))
    score = {"distance_right_hand" :distance_right_hand,
            "distance_right_foot" : distance_right_foot,
            "différence d'amplitude en x" : np.sqrt(np.sum(amplitude_x_parfaite - amplitude_x)) ** 2,
            "différence d'amplitude en y" : np.sqrt(np.sum(amplitude_y_parfaite - amplitude_y)) ** 2}
    
    return score 

# def comparaison_avec_course_parfaite(course, course_parfaite):

#     # definition des variables
#     x_parfaite_right_hand = course_parfaite[:,10,0]
#     y_parfaite_right_hand = course_parfaite[:,10,1]
#     x_parfaite_right_foot = course_parfaite[:,16,0]
#     y_parfaite_right_foot = course_parfaite[:,16,1]
    
#     amplitude_y_parfaite = np.sqrt(np.sum((min(y_parfaite_right_hand)- max(y_parfaite_right_hand)) ** 2))
#     amplitude_x_parfaite = np.sqrt(np.sum((min(x_parfaite_right_hand)- max(x_parfaite_right_hand)) ** 2))

#     x_right_hand = course[:,10,0]
#     y_right_hand = course[:,10,1]
#     x_right_foot = course[:,16,0]
#     y_right_foot = course[:,16,1]
    
#     amplitude_y = np.sqrt(np.sum((min(y_right_hand)- max(y_right_hand)) ** 2))
#     amplitude_x= np.sqrt(np.sum((min(x_right_hand)- max(x_right_hand)) ** 2))

#     y_parfaite_right_hand = extraire_points(x_parfaite_right_hand, y_parfaite_right_hand, x_right_hand, tolérance=0.02)
#     y_parfaite_right_foot = extraire_points(x_parfaite_right_foot, y_parfaite_right_foot, x_right_foot, tolérance=0.02)

#     def choose(y_parfaite,y):
#         y_parfaite_choose = y_parfaite[:,0]
#         for i in range(len(y)):
#             print(i)
#             if  np.abs(y_parfaite[i,1]-y[i]) < np.abs(y_parfaite[i,0]-y[i]):
#                 y_parfaite_choose[i] = y_parfaite[i,1]
#         return y_parfaite_choose

#     y_parfaite_right_hand = choose(y_parfaite_right_hand,y_right_hand)
#     y_parfaite_right_foot = choose(y_parfaite_right_foot,y_right_foot)

#     assert len(y_parfaite_right_hand) == len(x_right_hand)
#     assert len(y_parfaite_right_foot) == len(x_right_foot)

#     distance_right_hand = np.sqrt(np.sum((y_parfaite_right_hand - y_right_hand) ** 2))
#     distance_right_foot = np.sqrt(np.sum((y_parfaite_right_foot - y_right_foot) ** 2))
#     score = {"distance_right_hand" :distance_right_hand,
#             "distance_right_foot" : distance_right_foot,
#             "différence d'amplitude en x" : np.sqrt(np.sum(amplitude_x_parfaite - amplitude_x)) ** 2,
#             "différence d'amplitude en y" : np.sqrt(np.sum(amplitude_y_parfaite - amplitude_y)) ** 2}
    
#     fig, ax = plt.subplots()

#     # Tracer les points
#     ax.scatter(course_parfaite[:,16,0], course_parfaite[:,16,1], s=20, marker='.', color='blue', label='Pied droit Lemaitre')
#     ax.scatter(course_parfaite[:,10,0], course_parfaite[:,10,1], s=20, marker='.', color='red', label='Main droite Lemaitre')


#     ax.scatter(course[:,10,0], course[:,10,1], s=20, marker='.', color='orange', label='Main droite Course')
#     ax.scatter(course[:,16,0], course[:,16,1], s=20, marker='.', color='purple', label='Pied droit Course')

#     ax.legend()
#     # Ajouter des axes et des étiquettes
#     ax.axhline(y=0, color='black', linewidth=0.5)
#     ax.axvline(x=0, color='black', linewidth=0.5)
#     ax.set_xlabel('Coordonnée X')
#     ax.set_ylabel('Coordonnée Y')
#     ax.set_title('Analyse du mouvement (keypoints_lite_rel)')

#     # Améliorer la visibilité
#     plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
#     plt.tight_layout()

#     # Afficher le graphique
#     plt.show()
    
#     return score