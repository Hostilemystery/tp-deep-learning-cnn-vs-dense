import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👗",
    layout="wide"
)

# Titre et description améliorés
st.title("🔮 Fashion MNIST Classifier")
st.markdown("""
**Classifiez des images de vêtements avec nos modèles de Deep Learning!**
- 🧦 Modèle Dense: Réseau de neurones simple (précision ~90%)
- 🌀 Modèle CNN: Réseau convolutif avancé (précision ~94%)
""")

# Chargement des nouveaux modèles avec gestion d'erreur


@st.cache_resource(show_spinner="Chargement du modèle Dense...")
def load_dense_model():
    try:
        return tf.keras.models.load_model('./models/fashion_mnist_dense_v2.h5')
    except Exception as e:
        st.error(f"Erreur de chargement du modèle Dense: {str(e)}")
        return None


@st.cache_resource(show_spinner="Chargement du modèle CNN...")
def load_cnn_model():
    try:
        return tf.keras.models.load_model('./models/fashion_mnist_cnn_v2.h5')
    except Exception as e:
        st.error(f"Erreur de chargement du modèle CNN: {str(e)}")
        return None

# Interface de chargement d'image améliorée


def load_image():
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader(
            "Téléversez une image de vêtement (28x28px)",
            type=["png", "jpg", "jpeg"],
            key="uploader"
        )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('L')
            image = image.resize((28, 28))

            with col2:
                st.image(image, caption="Image téléversée", width=200)
                st.write(f"Format: {image.size}, Mode: {image.mode}")

            return image
        except Exception as e:
            st.error(f"Erreur de traitement d'image: {str(e)}")
            return None

# Préprocessing adaptatif selon le modèle


def preprocess_image(image, model_type):
    img_array = np.array(image)

    # Inversion des couleurs pour correspondre au dataset Fashion MNIST
    img_array = 255 - img_array  # Important!

    # Normalisation spécifique au modèle
    if model_type == "Dense":
        img_array = img_array.astype('float32') / 255.0
    else:  # CNN gère la normalisation via couche Rescaling
        img_array = img_array.astype('float32')

    return img_array.reshape(1, 28, 28, 1)

# Visualisation des prédictions améliorée


def plot_predictions(probs, classes):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    # Barres colorées par probabilité
    colors = plt.cm.viridis(probs * 0.8)
    bars = ax.bar(classes, probs, color=colors)

    # Annotation de la prédiction principale
    max_idx = np.argmax(probs)
    bars[max_idx].set_color('red')
    ax.text(max_idx, probs[max_idx]+0.02,
            f'{probs[max_idx]:.1%}',
            ha='center',
            color='darkred')

    plt.xticks(rotation=45)
    plt.ylim(0, 1.1)
    plt.title("Probabilités de prédiction", pad=20)
    plt.ylabel("Confiance")
    plt.tight_layout()
    return fig

# Interface principale


def main():
    # Chargement des modèles
    dense_model = load_dense_model()
    cnn_model = load_cnn_model()

    # Sélection du modèle
    model_type = st.radio(
        "Choisissez votre modèle:",
        ["Dense", "CNN"],
        horizontal=True,
        index=1
    )

    # Chargement d'image
    image = load_image()

    if image and st.button("🔎 Lancer la prédiction"):
        with st.spinner("Analyse en cours..."):
            # Préprocessing adaptatif
            processed_image = preprocess_image(image, model_type)

            # Sélection du modèle
            model = dense_model if model_type == "Dense" else cnn_model

            if model:
                try:
                    preds = model.predict(processed_image, verbose=0)[0]
                    classes = ["T-shirt", "Pantalon", "Pull", "Robe",
                               "Manteau", "Sandale", "Chemise", "Basket", "Sac", "Botte"]

                    # Affichage des résultats
                    st.success("Prédiction réussie!")

                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric(
                            label="Classe prédite",
                            value=classes[np.argmax(preds)],
                            delta=f"{np.max(preds):.1%} confiance"
                        )

                    with col2:
                        st.pyplot(plot_predictions(preds, classes))
                except Exception as e:
                    st.error(f"Erreur de prédiction: {str(e)}")


# Sidebar avec informations supplémentaires
st.sidebar.markdown("## ℹ️ À propos")
st.sidebar.markdown("""
Cette application utilise des réseaux de neurones pour classifier des images de vêtements selon 10 catégories:

1. **Modèle Dense**  
   - Architecture simple de type perceptron multicouche
   - Précision: ~90%
   
2. **Modèle CNN**  
   - Architecture convolutive avancée
   - Précision: ~94%
   
**Conseils d'utilisation:**
- Utilisez des images carrées (28x28px idéal)
- Fond clair pour meilleure détection
- Images contrastées
""")

# Footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    text-align: center;
    padding: 1rem;
    color: #666;
}
</style>
<div class="footer">
    Développé avec ❤️ par [Freddy Nanji] | 
    <a href="https://github.com/freddynanji/fashion-mnist-classifier">Code source</a>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
