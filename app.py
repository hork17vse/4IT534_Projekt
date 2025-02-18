import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import time
import os

# Konfigurace stránky
st.set_page_config(
    page_title="Skin Cancer AI",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cesta k modelu
MODEL_PATH = "model/best_resnet50.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Načtení modelu ResNet-50
model = models.resnet50()
num_classes = 7
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, num_classes)
)

# Načtení váh modelu
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    st.error("Modelové váhy nebyly nalezeny! Zkontrolujte cestu k souboru.")

model = model.to(device)
model.eval()

# Definice transformací
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Definice tříd
CLASS_NAMES = {
    0: "Benigní keratóza (bkl)",
    1: "Melanom (mel) ❗",
    2: "Nevus (nv)",
    3: "Dermatofibrom (df)",
    4: "Bazocelulární karcinom (bcc) ❗",
    5: "Aktinická keratóza / intraepidermální karcinom (akiec) ❗",
    6: "Vaskulární léze (vasc)"
}

# Sidebar s informacemi o modelu
st.sidebar.title("O modelu")
st.sidebar.info(
    """
    - **Použitý model:** ResNet-50
    - **Počet trénovacích epoch:** 10
    - **Přesnost na testovacích datech:** ~88%
    - **AUC na testovacích datech:** ~97%
    - **Použitý dataset:** HAM10000
    """
)

# Nadpis a popis aplikace
st.title("🔬 Diagnostika rakoviny kůže pomocí AI")
st.write("Nahrajte fotku kožní léze a zjistěte, zda se jedná o benigní nebo maligní nález.")

# Nahrání obrázku
uploaded_file = st.file_uploader("Nahrajte obrázek (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    input_image = transform(image).unsqueeze(0).to(device)

    with st.spinner("Analyzuji obrázek..."):
        time.sleep(2)
        with torch.no_grad():
            output = model(input_image)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()
            predicted_class = np.argmax(probabilities)

    # Zobrazení obrázku a výsledků
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Nahraný obrázek", use_column_width=True)

    with col2:
        diagnosis = CLASS_NAMES[predicted_class]
        prob_percentage = probabilities[0][predicted_class] * 100

        st.subheader("Výsledky predikce")
        if predicted_class in [1, 4, 5]:
            st.markdown(f'<span style="color:red; font-weight:bold;">{diagnosis} ({prob_percentage:.2f}%)</span>', unsafe_allow_html=True)
            st.warning("Model detekoval potenciálně nebezpečnou lézi! Doporučujeme konzultaci s dermatologem.")
        else:
            st.markdown(f'<span style="color:green; font-weight:bold;">{diagnosis} ({prob_percentage:.2f}%)</span>', unsafe_allow_html=True)
            st.success("Model neindikuje nebezpečnou lézi. Pokud máte pochybnosti, konzultujte s lékařem.")

        # Vizualizace pravděpodobností
        st.subheader("Pravděpodobnosti jednotlivých tříd:")
        for idx, prob in enumerate(probabilities[0]):
            st.progress(int(prob * 100))
            st.write(f"**{CLASS_NAMES[idx]}**: {prob:.2%}")
