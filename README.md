# Projet_hpe

Ce projet vise à mettre en place l'estimation de pose humaine en utilisant l'api de MMPose.

## Installation

Pour mettre en place un environnement propice à l'exécution des scripts, il faut suivre les étapes dans la documentation de [MMPose](https://mmpose.readthedocs.io/en/latest/installation.html). 

Étape 0. Téléchargez et installez Miniconda depuis le [site officiel](https://docs.conda.io/en/latest/miniconda.html).


Étape 1. Créez un environnement conda et activez-le.
```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```
Étape 2. Installez PyTorch.

```bash
#On GPU platform
conda install pytorch torchvision -c pytorch
#On CPU platforms:
conda install pytorch torchvision cpuonly -c pytorch
```

Étape 3. Installation de MMEngine, MMCV et MMDet à l'aide de MIM (OpenMMLab Model Dependency Installer) :

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
```

Étape 4. Clonez le dépôt projet_hpe et installez les dépendances

```bash
git clone https://github.com/atysp/projet_hpe.git
cd projet_hpe
pip install -r requirements.txt
```
Étape 5. Clonez le dépôt open-mmlab MMPose au sein du projet et installez les dépendances

```bash
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
touch __init__.py
cd ..
mim download mmpose --config rtmpose-l_8xb256-420e_aic-coco-256x192 --dest .
```

## Usage

Afin de tester l'installation de MMpose : 
```bash
cd mmpose
python demo/image_demo.py \
tests/data/coco/000000000785.jpg \
td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth \
--out-file vis_results.jpg \
--device cpu
ls
```
Normalement, un fichier nommé "vis_results.jpg" doit être présent dans le répertoire.

Vous pouvez ensuite tester le script python
"_application_lite/scripts/hpe.py" qui effectue l'inférence sur la vidéo choisie.

```bash
cd ..
python _application_lite/scripts/hpe.py
```

Lancez ensuite l'application Flask avec la commande : 

```bash
python _application_lite/main.py
```

Vous pouvez par exemple faire l'analyse des vidéos présentes dans le répertoire "projet_hpe/_videos" que vous avez cloné. 

## License

[MIT](https://choosealicense.com/licenses/mit/)