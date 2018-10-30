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

Esempio di struttura di una CNN.
![](images/CNN.png)

### 1. CNN from scratch
La CNN creata da 0 è formata da 15 strati con la seguente architettura:
1. Livello di input
2. Livello di Convoluzione 2-dim
3. Livello di Max Pooling 2-dim,  It partitions the input image into a set of non-overlapping rectangles and, for each such sub-region, outputs the maximum. 
4. Livello di passaggio per la funzione ReLU
5. Livello di convoluzione 2-dim
6. Livello di passaggio per la funzione ReLU
7. Livello di Avg Pooling
8. Livello di Convoluzione 2-dim
9. Livello di passaggio per la funzione ReLU
10. Livello di Avg Pooling
11. Livello Fully Connected
12. Livello di passaggio per la funzione ReLU
13. Livello Fully Connected
14. Livello per la funzione 'softmax'
15. Livello finale di classificazione che sfrutta la cross entropy

Per l'addestramento della CNN si sottopone alla rete neurale l'intero dataset mescolato più volte, dove il numero di "mescolamenti" è chiamato `MaxEpochs`.
Inoltre per ogni iterazione di allenamento il dataset è processato in maniera che un sottoinsieme del set di training, definito dalla variabile `MiniBatchSize` venga usato per valutare il gradiente della funzione di perdita e aggiornare i pesi.

Come funzione di errore viene utilizzata `sgdm`, ovvero *Stochastic Gradient Descent with Momentum optimizer*, una funzione che esegue la discesa gradiente stocastica con un'ottimizzazione del momento che, in questo caso è stato lasciato di default.
L'algoritmo di discesa gradiente aggiorna i parametri del network (pesi e bias, definiti nell'architettura) per minimizzare la funzione di perdita prendendo piccoli steps nella direzione del gradiente negativo della perdita, definiti da:

<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_{l&plus;1}&space;=&space;\theta_{l}-\alpha&space;\Delta&space;E(\theta_l)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_{l&plus;1}&space;=&space;\theta_{l}-\alpha&space;\Delta&space;E(\theta_l)" title="\theta_{l+1} = \theta_{l}-\alpha \Delta E(\theta_l)" /></a>

dove l è il numero di iterazioni, <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\alpha&space;>0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\alpha&space;>0" title="\alpha >0" /></a> è la frequenza di apprendimento, <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta" title="\theta" /></a> è il vettore parametro e <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;E(\theta)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;E(\theta)" title="E(\theta)" /></a> è la funzione di perdita. 
L'algoritmo di discesa gradiente stocastica valuta il gradiente e aggiorna i parametri usando un subset del training set, chiamato mini-batch. Ad ogni iterazione, cioè ogni valutazione del gradiente usando il mini-batch, l'algoritmo fa un passo avanti nella minimizzazione la funzione di perdita. L'intero passo dell'algoritmo di training sull'intero set di training usando le mini-batches è un epoca-

All'algoritmo di discesa stocastica viene aggiunto un termine di momento per ridurre l'oscillazione lungo il cammino di discesa profonda verso il massimo. In questo caso, l'equazione che governa questo processo è la seguente:

<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_{l&plus;1}&space;=&space;\theta_l-\alpha&space;\Delta&space;E(\theta_l)&space;&plus;\gamma(\theta_l-\theta_{l-1})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_{l&plus;1}&space;=&space;\theta_l-\alpha&space;\Delta&space;E(\theta_l)&space;&plus;\gamma(\theta_l-\theta_{l-1})" title="\theta_{l+1} = \theta_l-\alpha \Delta E(\theta_l) +\gamma(\theta_l-\theta_{l-1})" /></a>

dove <a href="http://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{120}&space;\gamma" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\inline&space;\dpi{120}&space;\gamma" title="\gamma" /></a> determina il contributo del precedente step di gradiente all'iterazione corrente.  Inoltre, si è specificata anche la frequenza di apprendimento iniziale, tramite il parametro `InitialLearningRate`.

Di seguito è riportato lo script relativo al training:
```matlab
  varSize = 32;
  conv1 = convolution2dLayer(5,varSize,'Padding',2,'BiasLearnRateFactor',2);
  conv1.Weights = gpuArray(single(randn([5 5 3 varSize])*0.0001));
  fc1 = fullyConnectedLayer(64,'BiasLearnRateFactor',2);
  fc1.Weights = gpuArray(single(randn([64 576])*0.1));
  fc2 = fullyConnectedLayer(2,'BiasLearnRateFactor',2);
  fc2.Weights = gpuArray(single(randn([2 64])*0.1));

  layers = [
      imageInputLayer([varSize varSize 3], 'Name', 'input');
      conv1;
      maxPooling2dLayer(3,'Stride',2, 'Name', 'max_pool');
      reluLayer('Name', 'relu_1');
      convolution2dLayer(5,32,'Padding',2,'BiasLearnRateFactor',2, 'Name', 'conv_2');
      reluLayer('Name', 'relu_2');
      averagePooling2dLayer(3,'Stride',2, 'Name', 'avg_pool_1');
      convolution2dLayer(5,64,'Padding',2,'BiasLearnRateFactor',2, 'Name', 'conv_3');
      reluLayer('Name', 'relu_3');
      averagePooling2dLayer(3,'Stride',2, 'Name','avg_pool_2');
      fc1;
      reluLayer('Name', 'relu_4');
      fc2;
      softmaxLayer('Name', 'softmax');
      classificationLayer('Name', 'classification')];

  opts = trainingOptions('sgdm', ...
      'InitialLearnRate', 0.001, ...
      'L2Regularization', 0.004, ...
      'MaxEpochs', 10, ...
      'Shuffle','every-epoch', ...
      'MiniBatchSize', 10, ...
      'ValidationData',imdsValidation, ...
      'ValidationFrequency',350, ...
      'Verbose', true, ...
      'VerboseFrequency', 350, ...
      'Plots','training-progress', ...
      'ExecutionEnvironment', 'auto');

  [net, info] = trainNetwork(imdsTrain, layers, opts);

  cifar10_net= net;
  save cifar10_net
```

I risultati sono visualizzati in tempo reale in Training Progress e sono mostrati nella seguente figura:

Per quanto riguarda il testing si è verificato che il valore dell'accuratezza fosse compatibile con quello della validazione ottenuta durante il processo di training. Una volta classificate, le immagini vengono mostrate con la relativa label contrassegnata con il colore rosso se è errata e verde se è giusta. Inoltre, l'accuratezza di testing è fornita dalla media tra i termini della diagonale della matrice di confusione ciascuno normalizzato per il numero totale di esempi di training.
Di seguito è riportato lo script relativo al testing:
```matlab
  da TrainingCIFAR10.m
  load cifar10_net;
  rootFolder2 = 'cifar10/cifar10Test';
  imdsTest = imageDatastore(fullfile(rootFolder2, categories), ...
      'LabelSource', 'foldernames');
  imdsTest.ReadFcn = @readFunctionTrain;

  labels = classify(cifar10_net, imdsTest);

  for i = 1:50
      ii = randi(2000);
      im = imread(imdsTest.Files{ii});
      imshow(im);
      if labels(ii) == imdsTest.Labels(ii)
         colorText = 'g'; 
      else
          colorText = 'r';
      end
      title(char(labels(ii)),'Color',colorText);
  end

  % This could take a while if you are not using a GPU
  confMat = confusionmat(imdsTest.Labels, labels);
  confMat2 = confMat./sum(confMat,2);
  mean(diag(confMat2))
```

Di seguito sono riportate 10 immagini risultanti dal testing di CIFAR10:
![](images/CIFAR10/Testing1.png) ![](images/CIFAR10/Testing2.png) ![](images/CIFAR10/Testing3.png) ![](images/CIFAR10/Testing4.png) ![](images/CIFAR10/Testing5.png) ![](images/CIFAR10/Testing6.png) ![](images/CIFAR10/Testing7.png) ![](images/CIFAR10/Testing8.png) ![](images/CIFAR10/Testing9.png) ![](images/CIFAR10/Testing10.png)
