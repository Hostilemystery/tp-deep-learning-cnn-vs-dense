# Deep Learning Project - CNN vs Dense Networks for Fashion MNIST

## Description

Ce projet compare les performances de réseaux de neurones denses et convolutifs (CNN) sur le dataset Fashion MNIST. L'application inclut :

- 🧠 Deux modèles entraînés (Dense et CNN)
- 📊 Visualisation des résultats avec TensorBoard
- 🚀 Interface de prédiction avec Streamlit

## Dataset Requirements

### Téléchargement des données

1. Téléchargez les fichiers du dataset Fashion MNIST :

   - [Fashion MNIST] (https://www.kaggle.com/datasets/zalando-research/fashionmnist/data)

2. Créez la structure de dossier requise :

   ```bash
   mkdir -p data
   ```

3. Placez les fichiers téléchargés dans le dossier data/ :

   ```bash
   mv fashion-mnist_train.csv data/
   mv fashion-mnist_test.csv data/
   ```

## Installation

### Environnement Virtuel (Recommandé)

1. Création de l'environnement :

   ```bash
   # Avec venv (Python 3.3+)
   python -m venv myenv

   # Ou avec virtualenv
   # virtualenv myenv
   ```

2. Activation de l'environnement :

   ```bash
   # Linux/MacOS
   source myenv/bin/activate

   # Windows
   .\myenv\Scripts\activate
   ```

3. Installation des dépendances :
   ```bash
   pip install -r requirements.txt
   ```

### Installation sans Environnement Virtuel

```bash
pip install --user -r requirements.txt
```

## Fichier Requirements

Le fichier `requirements.txt` contient :

```
pandas ==2.2.3
seaborn == 0.13.2
matplotlib ==3.10.1
tensorflow ==2.19.0
mlflow == 2.21.0
streamlit == 1.43.2
ipykernel
```

## Utilisation

1. Visualisation et Entraînement des modèles :

   a. Le fichier : exploration_fashion_mnist.ipynb

   b. Le fichier : comparaison_modeles.ipynb

2. Visualisation avec TensorBoard et mlflow:

   ```bash
   tensorboard --logdir=logs
   mlflow ui
   ```

3. Lancement de l'interface Streamlit :

   ```bash
   streamlit run streamlit_app.py
   ```
