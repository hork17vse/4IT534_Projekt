
# Rozpoznávání rakoviny kůže pomocí AI (ResNet18)  

Tento projekt obsahuje **konvoluční neuronovou síť (CNN)** určenou k **detekci a klasifikaci kožních lézí** z obrázků. Model je schopen rozpoznat **různé typy kožních onemocnění** a určit, zda jde o **benigní** nebo **maligní** útvar.  

📌 **Dataset:** [HAM10000 – Skin Cancer MNIST](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data)  
📌 **Cíl projektu:** Vytvořit přesný a spolehlivý model pro automatickou analýzu kožních lézí, který může pomoci lékařům při diagnostice rakoviny kůže.  
📌 **Output:** **Streamlit aplikace**, do které lze nahrát obrázek a model provede **automatické vyhodnocení léze**.  

---

## 📂 **Struktura projektu**  

- `data/` – **Složka pro dataset** (*uživatel musí stáhnout data z Kaggle, vytvořit složku data a umístit je tam*)  
- `model/` – **Finální model** vygenerovaný skriptem `training.py`, složka se vytvoří automaticky po spuštění scriptu training.py  
- `app.py` – **Streamlit aplikace**, umožňuje nahrát obrázek a model provede vyhodnocení  
- `legacy_training.py` – **První iterace trénování modelů**, použita pro srovnání a výběr finálního modelu  
- `training.py` – **Hlavní skript pro trénování modelu**  
- `requirements.txt` – **Seznam základních závislostí** pro běh modelu a aplikace  
- `requirements_full.txt` – **Seznam všech závislostí**, včetně podpory pro analýzu starších modelů  

---

## ⚙️ **Instalace závislostí**  

Můžeš si vybrat mezi **aktuální verzí** (pouze pro trénování a Streamlit aplikaci) nebo **plnou verzí** (včetně podpory pro starší modely).  

### 1️⃣ **Aktuální verze** *(doporučeno, pokud chceš jen trénovat model a spustit Streamlit aplikaci)*  
Nainstaluje pouze základní závislosti pro **training.py** a **app.py**:  

```bash
pip install -r requirements.txt
```

---

### 2️⃣ **Plná verze** *(pokud chceš pracovat i s legacy_training.py a provádět analýzy modelů)*  
Nainstaluje **všechny balíčky** potřebné pro trénování modelu, starší iterace i vizualizaci výsledků:  

```bash
pip install -r requirements_full.txt
```

---

## 🚀 **Jak spustit projekt**  

### 1️⃣ **Klonování repozitáře**  
```bash
git clone https://github.com/hork17vse/4IT534_Projekt.git
```

### 2️⃣ **Instalace závislostí**  
Vyber si aktuální nebo plnou verzi (viz výše).  

### 3️⃣ **Stažení datasetu**  
Dataset stáhni z Kaggle: [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data)  
a umísti ho do složky `data/`.  

### 4️⃣ **Spuštění trénování modelu**  
```bash
python training.py
```

### 5️⃣ **Spuštění Streamlit aplikace**  
```bash
streamlit run app.py
```

💡 Po spuštění se otevře webová aplikace, kde můžeš **nahrát obrázek** a model provede **vyhodnocení léze**.

---

## 🛠 **Použité technologie**  

- **Python**  
- **TensorFlow / Keras**  
- **PyTorch**  
- **OpenCV**  
- **NumPy & Pandas**  
- **Matplotlib / Seaborn**  
- **Streamlit**  

---

## 📌 **Možné vylepšení**  

- Přidání datové augmentace  
- Optimalizace hyperparametrů  
- Přidání vizualizace heatmap pro interpretaci modelu  
- Nasazení modelu jako online webové aplikace  

---
