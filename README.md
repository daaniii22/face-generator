# **Generación de Caras con Autoencoders**

Este repositorio se centra en el entrenamiento e inferencia de Autoencoders (AE) convolucionales para la generación de nuevas caras humanas y su evaluación en la competición de Kaggle (`mg-practica-1-generando-caras`).

---

## **1. Objetivo**

A partir de un subconjunto de **30.000 imágenes de CelebAMask-HQ** redimensionadas a `64x64` píxeles en escala de grises, el objetivo es entrenar modelos generativos capaces de sintetizar **200 caras nuevas** (formato de entrega CSV de `200 × 4097`: columna `ID` + `4096` píxeles) cuya distribución se aproxime lo máximo posible a las imágenes reales, evaluándose mediante la métrica **FID (Fréchet Inception Distance)**.

El reto principal de utilizar un Autoencoder determinista como generador es que **un AE aprende a comprimir y reconstruir, pero no impone una distribución conocida sobre su espacio latente**. Por tanto, la calidad de las muestras generadas depende tanto de la capacidad de reconstrucción de la red como de la estrategia empleada para muestrear el espacio latente.

---

## **2. Metodología común**

Para que todos los experimentos sean estrictamente comparables, todos los notebooks comparten la misma configuración base:
- **Partición de datos:** `SEED = 1234`, split determinista `90%` entrenamiento (`27.000` imágenes) y `10%` validación (`3.000` imágenes).
- **Función de pérdida:** combinación `0.5 · L1 + 0.5 · (1 - SSIM)`, buscando preservar tanto la nitidez de bordes a nivel de píxel (L1) como la estructura y el contraste (SSIM).
- **Entrenamiento:** optimizador Adam (`1e-3`), *data augmentation* con volteo horizontal aleatorio en entrenamiento (`tf.image.random_flip_left_right`), y callbacks `ReduceLROnPlateau` + `EarlyStopping`.
- **Evaluación FID local:** calculado con `FrechetInceptionDistance(feature=2048)` sobre `x_val` tanto con `N = 200` (tamaño real de la submission) como con `N = 512` (estimación más estable). Además, se mide el **FID de las reconstrucciones** de validación como techo teórico del decoder.

---

## **3. Narrativa y orden de lectura de los notebooks**

El proyecto está diseñado como una serie incremental. Para entender la evolución de las decisiones de diseño y los resultados, **los notebooks deben leerse en el siguiente orden**:

#### 1. `AE_complex1.ipynb` — *Modelo Baseline*
Establece el punto de partida de la serie con una arquitectura sencilla y un método directo para poder muestrear con una normal estándar.

#### 2. `AE_complex2.ipynb` — *Escalado de arquitectura y batería de muestreo sobre el latente*
Ataca los dos cuellos de botella identificados en el baseline (capacidad de reconstrucción y pérdida por muestreo).

#### 3. `overfitting.ipynb` — *Sensibilidad a la semilla*
Dado que el FID evaluado sobre solo `200` imágenes tiene varianza muestral se prueba la sensibilidad con el mismo modelo y varias semillas para ver el resultado en competición.

#### 4. `fusion_caras.ipynb` — *Interpolación de caras*
Carga `encoder_complex2.keras` y `decoder_complex2.keras` desde `checkpoints/` para fusionar dos fotografías en el espacio latente. Este notebook es únicamente de inferencia y no vuelve a entrenar los modelos.

El enlace a los modelos entrenados es el siguiente: [Modelos entrenados](https://drive.google.com/drive/folders/1MU8rvtOZjd-GhR4rIhM1BbAh01GR0SW4?usp=share_link).
