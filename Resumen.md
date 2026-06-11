# Clasificador CNN de Lesiones Dermatológicas — Resumen

## Contexto

Se retoma el problema de clasificación de 9 clases de lesiones dermatológicas del proyecto anterior (MLP), con el objetivo de superar el **60.95% de accuracy en test** usando CNNs.

---

## Flujo del Proyecto

### 0. EDA
Sin cambios respecto al proyecto anterior. El split 60/20/20 estratificado y la limpieza de duplicados por MD5 ya estaban correctos en el notebook 1 (se hacen en tiempo de ejecución, no en el EDA).

---

### 1. CNN Simple — Prueba Manual (`1_CNN_Simple.ipynb`)

#### Arquitectura: AlexNet-like
Se diseñó una CNN inspirada en AlexNet (Krizhevsky et al., 2012), adaptada a imágenes pequeñas:

| Capa | Detalle |
|---|---|
| Conv1 (3→32, 3×3) + BN + ReLU + MaxPool | Extracción de bordes y texturas básicas |
| Conv2 (32→64, 3×3) + BN + ReLU + MaxPool | Formas simples |
| Conv3 (64→128, 3×3) + BN + ReLU | Features de nivel medio |
| Conv4 (128→128, 3×3) + BN + ReLU | Features más abstractas |
| Conv5 (128→64, 3×3) + BN + ReLU + MaxPool | Features de alto nivel |
| FC(512) + Dropout + FC(256) + Dropout + FC(9) | Clasificador |

**Diferencias con AlexNet original**: kernels 3×3 en lugar de 11×11 (las imágenes son 64px, no 224px), LRN reemplazada por BatchNorm (más estable), escala de filtros reducida proporcionalmente.

#### Experimentos realizados (en orden incremental)

| # | Qué se cambió | Hipótesis |
|---|---|---|
| 1 | Base sin augmentations | CNN debería superar al MLP sin augmentations por su invarianza traslacional |
| 2 | + HFlip + VFlip | Las lesiones no tienen orientación canónica → flips son válidos semánticamente |
| 3 | + RandomBrightnessContrast + CLAHE | Variaciones de iluminación son frecuentes en imágenes médicas |
| 4 | + HueSaturationValue + Rotate | Diferencias de tono de piel y ángulo de captura |
| 5 | + Weight Decay 1e-4 | Regularización L2 para reducir overfitting |
| 6 | Dropout 0.5 → 0.3 | Dropout muy alto puede limitar capacidad en red no tan grande |
| 7 | Momentum 0.9 → 0.99 | En proyecto MLP, 0.99 fue consistentemente mejor |
| 8 | Batch 32 → 16 | En proyecto MLP, batch pequeño fue consistentemente mejor |

> Completar con resultados reales tras la ejecución.

#### Transfer Learning — ResNet18 (Bonus)

Se aplicó transfer learning en 2 fases:
1. **Feature extraction**: backbone congelado, solo se entrena el clasificador FC (LR=1e-3, Adam)
2. **Fine-tuning**: se descongelan `layer3`, `layer4` + FC (LR=1e-4, SGD con momentum)

**Justificación según teoría de TL**: el dataset es de tamaño mediano y el dominio es diferente a ImageNet (lesiones de piel vs fotos naturales) → se prefiere congelar las primeras capas (que detectan bordes/texturas universales) y reentrenar las capas más profundas que codifican features de alto nivel específicos del dominio.

---

### 2. Búsqueda de Hiperparámetros (`2_CNN_Busqueda_HP.ipynb`)

#### Espacio explorado

- `input_size`: 32, 64
- `batch_size`: 16, 32
- `lr`: 1e-3, 1e-4
- `optimizer`: SGD, Adam
- `momentum`: 0.9, 0.99
- `weight_decay`: 0, 1e-4, 1e-3
- `dropout`: 0.0, 0.2, 0.3, 0.5
- Probabilidades de augmentations: HFlip, VFlip, RBContrast, CLAHE, HSV, Rotate

**Total**: ~6144 combinaciones → se muestreó ~5% (~307 corridas) con Random Search.

#### Conclusiones de la búsqueda

> Completar con resultados reales tras la ejecución.

Preguntas a responder:
1. ¿SGD o Adam funcionó mejor?
2. ¿Batch 16 o 32?
3. ¿Input 32 o 64? (Para CNN se espera que 64 gane, a diferencia del MLP)
4. ¿Qué augmentations aportaron más?
5. ¿Qué dropout fue más efectivo?

---

## Registro de Decisiones

| Decisión | Justificación |
|---|---|
| AlexNet-like en lugar de VGG/ResNet desde cero | AlexNet es más simple, adecuada para datasets pequeños-medianos, y es la topología recomendada en el enunciado |
| BatchNorm en lugar de LRN | BN es más estable, acelera entrenamiento y regulariza mejor que LRN (standard en 2024) |
| Kernels 3×3 en lugar de 11×11 | Las imágenes de 64px no tienen suficiente resolución para un receptive field tan grande |
| Normalización ImageNet (mean/std) | Estándar para modelos preentrenados; también mejora convergencia en CNNs propias |
| Oversampling solo en train | Balancear en val/test inflaría las métricas artificialmente |
| Transfer Learning con ResNet18 en lugar de VGG16 | ResNet18 es más liviana, suficiente para este dataset, y más fácil de fine-tunear |
| Test comentado hasta el final | Regla fundamental: el test set se toca una sola vez para no contaminar la selección de HPs |

---

## Resultado Final

> Completar con accuracy en test del modelo campeón.

| Modelo | Val Acc | Test Acc |
|---|---|---|
| MLP (proyecto anterior) | 65.58% | 60.95% |
| CNN AlexNet-like (mejor manual) | — | — |
| CNN AlexNet-like (mejor RS) | — | — |
| ResNet18 fine-tuning | — | — |
| **Campeón** | — | — |
