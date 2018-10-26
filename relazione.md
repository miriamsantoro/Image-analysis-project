# Confronto di CNN per la classificazione di cani e gatti
## Elaborato per il corso di Analisi delle Immagini
### Miriam Santoro
#### aa. 2017/2018, Università di Bologna - Corso di laurea magistrale in Fisica Applicata

## Scopo del progetto
Il presente lavoro è stato realizzato in linguaggio `matlab` e si pone come obiettivo l'implementazione e la valutazione di:
1. Rete neurale convoluzionale (CNN) da zero;
2. Rete neurale convoluzionale (CNN) da zero + Augment del dataset;
3. Modello di CNN pre-allenato (`AlexNet`).

Queste tre casistiche sono state applicate a due dataset diversi, ovvero:
1. `CIFAR10`
2. `Kaggle`

Il progetto ha previsto anche l'implementazione di add-ons per matlab, quali:
1. `Image Processing Toolbox`
2. `Neural Network Toolbox`
3. `Parallel Computing Toolbox`
4. `Neural Network Toolbox Model for AlexNet Network`
5. `Deep Learning Network Analyzer for Neural Network Toolbox`

## Esecuzione del progetto
Le reti neurali sono state implementate in 6 script diversi, a seconda della loro tipologia e del dataset utilizzato. Nello specifico:
- `TrainingCIFAR10.m` contiene la CNN allenata da zero sul dataset CIFAR10;
- `Training.m` contiene la CNN allenata da zero sul dataset Kaggle;
- `Training_AugmentCIFAR10.m` contiene la CNN allenata da 0 sulle immagini augmented del dataset CIFAR10;
- `Training_Augment.m` contiene la CNN allenata da 0 sulle immagini augmented del dataset Kaggle;
- `TrasfLearning_CIFAR10.m` contiene la CNN AlexNet pre-allenata adattata alle immagini del dataset CIFAR10;
- `TrasfLearning.m` contiene la CNN AlexNet pre-allenata adattata alle immagini del dataset Kaggle.

## Dataset (Note preliminari)
### CIFAR10
Il dataset CIFAR-10 è formato da 60000 immagini colorate 32x32 disposte in 10 classi. Quindi, ci sono 6000 immagini per classe che sono divise in 5000 immagini di training e 1000 immagini di testing.

In questo progetto vengono usate solo 2 delle 10 classi appartenenti a questo dataset: `Cat` e `Dog`.

Per scaricare il dataset e prepararlo in maniera da disporre le immagini in apposite cartelle è stato utilizzato lo script `DownloadCIFAR10.m`, scaricato dal sito di matlab (link: https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/62990/versions/3/previews/DeepLearningDemos/DownloadCIFAR10.m/index.html).

### Kaggle



