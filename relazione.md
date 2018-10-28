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
Di seguito si descrivono i dataset utilizzati per ogni CNN. Entrambi i dataset sono stati divisi in 90% training e 10% testing, supponendo che il numero di immagini per classe sia 60000, per evitare errori di overfitting.
Inoltre le 5000 immagini di training sono state ulteriormente divise in 70% training e 30% validation per regolare l'architettura del classificatore stesso, aggiustandone i parametri.

Per adattare i dataset alle dimensioni di input delle CNN sono usati gli script:
- `readFunctionTrain` per la CNN da 0 e la CNN con l'augment. Questa funzione è in grado di ridimensionare ogni immagine di input in maniera da avere un'immagine di output di dimensione 32x32;
- `readFunctionTrain2` per la CNN basata su AlexNet. Questa funzione è in grado di ridimensionare ogni immagine di input in maniera da avere un'immagine di output di dimensione 227x227.

In realtà ogni immagine ha dimensioni [n_pixel x n_pixel x 3] in quanto è un'immagine a colori.
### CIFAR10
Il dataset CIFAR-10 è formato da 60000 immagini colorate 32x32 disposte in 10 classi. Quindi, ci sono 6000 immagini per classe che sono divise in 5000 immagini di training e 1000 immagini di testing.

In questo progetto vengono usate solo 2 delle 10 classi appartenenti a questo dataset: `Cat` e `Dog`.

Per scaricare il dataset e prepararlo in maniera da disporre le immagini in apposite cartelle è stato utilizzato lo script `DownloadCIFAR10.m`, scaricato dal sito di matlab (link: https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/62990/versions/3/previews/DeepLearningDemos/DownloadCIFAR10.m/index.html).

I dati sono importati come mostrato nel seguente esempio di codice:
```matlab
  #from TrainingCIFAR10.m
  categories = {'Dog','Cat'};

  rootFolder = 'cifar10/cifar10Train';
  imds = imageDatastore(fullfile(rootFolder, categories), ...
      'LabelSource', 'foldernames');
  imds.ReadFcn = @readFunctionTrain;
  
  [imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');     #divide train set e validation set
```

### Kaggle
Il dataset Kaggle è formato da 25000 immagini colorate di dimensioni diverse disposte in due classi (ovvero `cats` e `dogs`). Quindi per ogni classe si hanno a disposizione 12500 immagini totali.

In questo progetto vengono usate solo 5000 immagini di training e 1000 immagini di testing per ogni classe. Le immagini di training sono le prime 5000, quelle di testing sono tra 5000 e 6000. 

Il dataset è stato scaricato dal sito della Kaggle e le immagini sono state poste manualmente nelle rispettive cartelle.

I dati sono importati come mostrato nel seguente esempio di codice:
```matlab
  #from Training.m
  categories = {'dogs','cats'};

  rootFolder = 'dataset/train_set';
  imds = imageDatastore(fullfile(rootFolder, categories), ...
      'LabelSource', 'foldernames');
  imds.ReadFcn = @readFunctionTrain;

  [imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
```


## Analisi delle CNN
Le reti neurali convoluzionali (CNN) hanno architetture che le rendono particolarmente adatte al riconoscimento delle immagini, in maniera da poterle classificare. Nello specifico, i diversi layer della rete neurale imparano a rilevare e identificare le diverse features delle immagini. A questi layer si aggiungono il penultimo layer che genera un vettore delle stesse dimensioni del numero di classi che la rete deve essere in grado di prevedere e l'ultimo layer che fornisce l'output di classificazione.

Siccome le CNN sono addestrate su molte immagini e lavorano con una grande quantità di dati e con diverse architetture, si ritiene opportuno utilizzare la GPU in quanto fondamentale per velocizzare significativamente il tempo necessario ad allenare un modello.

### 1. CNN from scratch




