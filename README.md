# Clustering non supervisionato del dataset Iris mediante implementazione manuale dell'algoritmo K-means

Progetto didattico Python per comprendere, eseguire e presentare K-means. Il nucleo dell'algoritmo è scritto da zero con **NumPy**. Scikit-learn serve soltanto a caricare Iris, standardizzare i dati e calcolare silhouette e ARI; nessuna implementazione esterna di K-means viene utilizzata, neppure nei test.

## Avvio rapido e Visual Studio Code

È richiesto Python **3.10 o successivo**. Aprire questa cartella in VS Code (`File > Apri cartella`), poi aprire un terminale PowerShell nella cartella del progetto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Nel progetto consegnato `.venv` è già stato preparato nell'ambiente corrente; è comunque possibile ricrearlo seguendo i comandi sopra. Un ambiente virtuale non è portabile tra computer.

Con l'estensione Python di VS Code, scegliere `Python: Select Interpreter` dalla tavolozza dei comandi e selezionare `.venv\Scripts\python.exe`. Il comando **Run Python File in Terminal** su `main.py` esegue l'intero progetto. Se si preferisce attivare l'ambiente:

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
python -m unittest discover -s tests -v
```

Se PowerShell impedisce l'attivazione, utilizzare direttamente `.venv\Scripts\python.exe`, come nel primo esempio; non occorre cambiare le impostazioni del sistema. Su Linux/macOS usare `python3 -m venv .venv` e `.venv/bin/python` al posto del percorso Windows.

Non sono richiesti input interattivi né download del dataset: Iris è incluso in scikit-learn. Le cartelle mancanti sono create automaticamente. I percorsi degli output sono calcolati dalla posizione di `main.py`, quindi l'esecuzione funziona anche da un'altra directory. Ogni avvio rigenera gli stessi file di risultato. I grafici sono salvati su disco senza aprire finestre.

## Apprendimento non supervisionato

Nel Machine Learning supervisionato si forniscono esempi con una risposta attesa. Nel clustering si cercano gruppi nei soli vettori di osservazione, senza fornire risposte corrette all'algoritmo. K-means raggruppa osservazioni vicine nello spazio delle feature.

In questo progetto `fit(X)` riceve esclusivamente una matrice numerica. Le specie reali non entrano nell'inizializzazione, negli aggiornamenti o nella scelta della migliore esecuzione: vengono usate soltanto per valutare e illustrare i risultati finali. Sapere che Iris contiene tre specie aiuta a progettare l'esperimento K=3, ma non fornisce le risposte al modello.

## Dataset e vettori di feature

`sklearn.datasets.load_iris()` fornisce **150 fiori**, con 50 osservazioni per specie: Iris setosa, Iris versicolor e Iris virginica. Ogni fiore è descritto dal vettore in centimetri

\[
x_i = (\text{lunghezza sepalo},\ \text{larghezza sepalo},\ \text{lunghezza petalo},\ \text{larghezza petalo}) \in \mathbb{R}^4.
\]

La matrice `X` ha forma `(150, 4)`; `y` ha forma `(150,)` e contiene i codici delle specie. Per esempio, il primo vettore originale è `(5.1, 3.5, 1.4, 0.2)`. Il dataset proviene dalla copia distribuita con scikit-learn, caricata senza modificare o filtrare campioni.

`data/iris_features.csv` conserva le quattro misure originali; `data/iris_labels.csv` conserva separatamente codice e nome della specie. La colonna `campione_id` permette di ricongiungere i file mantenendo la corrispondenza tra righe; **non è una feature**. La versione di scikit-learn è registrata in `metrics.json` per documentare la provenienza effettiva.

## Standardizzazione

Per ciascuna feature si applica

\[
z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}.
\]

`StandardScaler` calcola media e deviazione standard sull'intero dataset, usando la varianza di popolazione (`ddof=0`). Le colonne diventano centrate e hanno varianza unitaria. Anche se tutte le misure sono in centimetri, le loro dispersioni differiscono: senza standardizzazione le feature più disperse possono dominare la distanza.

L'esperimento è descrittivo sull'intero Iris: non si sta stimando la capacità di classificare un insieme di test. Per nuovi campioni si deve applicare **lo stesso scaler già addestrato**, mai riaddestrarlo sui nuovi dati. `predict()` riceve dati nello stesso spazio di `fit()`, quindi qui dati standardizzati. Media e scala sono salvate nel JSON. La standardizzazione modifica la geometria e quindi anche il clustering.

## Distanza euclidea e funzione obiettivo

Tra un campione standardizzato e un centroide:

\[
d(z_i, c_k) = \sqrt{\sum_{j=1}^{4}(z_{ij}-c_{kj})^2}.
\]

L'algoritmo usa direttamente le distanze **al quadrato**: la radice è crescente, quindi l'indice del minimo è lo stesso. Si risparmia così un'operazione senza cambiare l'assegnazione.

La funzione obiettivo è la somma degli errori quadratici (SSE), chiamata anche inerzia:

\[
J=\sum_{i=1}^{N}\lVert z_i-c_{\ell_i}\rVert_2^2,
\qquad \ell_i=\operatorname*{argmin}_{k}\lVert z_i-c_k\rVert_2^2.
\]

Una SSE minore indica gruppi più compatti secondo questa geometria. Non misura direttamente la corrispondenza con le specie. I valori di SSE qui si riferiscono alle quattro feature standardizzate, non a centimetri quadrati nelle unità originali.

## K-means passo per passo

1. Si scelgono casualmente K **indici distinti** dei campioni con `default_rng(seed).choice(..., replace=False)`. Questi vettori costituiscono i centroidi iniziali. Indici distinti possono contenere valori uguali se il dataset ha duplicati.
2. Si calcola la matrice `(N, K)` delle distanze quadratiche con il broadcasting di NumPy.
3. Ogni osservazione viene assegnata al centroide più vicino; in caso di parità `argmin` sceglie il primo indice.
4. Ogni centroide non vuoto viene sostituito dalla media dei membri del proprio cluster. La media minimizza la somma delle distanze quadratiche per assegnazioni fissate.
5. Per ogni cluster vuoto si sceglie il campione con maggiore distanza quadratica dal centroide al quale era assegnato prima dell'aggiornamento. Nella stessa iterazione si usano indici distinti per i recuperi. Solo i centroidi vuoti vengono ricollocati.
6. Si ricalcolano assegnazioni e SSE rispetto ai nuovi centroidi e si salva lo stato.
7. Si termina se le assegnazioni non cambiano, se la norma complessiva degli spostamenti è al massimo `tol=1e-6`, oppure dopo `max_iter=300` aggiornamenti.

La norma degli spostamenti è `np.linalg.norm(new_centers - centers)`, ossia la radice della somma dei quadrati di tutti gli spostamenti nelle quattro coordinate di tutti i centroidi (norma di Frobenius). `converged_` distingue l'arresto per convergenza dall'esaurimento delle iterazioni.

La media dei cluster non vuoti non aumenta la SSE per le vecchie assegnazioni; i centroidi vuoti non contribuivano a quella somma. La successiva assegnazione al minimo non può aumentarla. Per questa strategia anche il recupero dei cluster vuoti preserva la non crescita, salvo arrotondamenti numerici. Se i dati contengono meno di K vettori distinti, possono comunque restare cluster senza membri: non si forzano assegnazioni incompatibili con il centroide più vicino.

**Coerenza finale:** `labels_` è sempre l'assegnazione al più vicino dei centroidi effettivamente restituiti e `inertia_` è calcolata su quella coppia. Con un arresto anticipato per tolleranza o limite di iterazioni, i centroidi potrebbero non coincidere ancora con le medie delle nuove assegnazioni; `predict(X)` rimane comunque coerente con `labels_`.

`inertia_history_` e `centroids_history_` includono l'inizializzazione all'indice 0 e uno stato per ogni aggiornamento: la lunghezza è `n_iter_ + 1`. Gli array dei centroidi vengono copiati per evitare di modificare retroattivamente la cronologia.

## Venti inizializzazioni e scelta di K

K-means può fermarsi in minimi locali. `fit_best_kmeans` genera **20 seed distinti** a partire dal seed generale 42, esegue 20 addestramenti indipendenti e restituisce quello con inerzia finale minima, senza consultare le specie. Le inerzie e i seed dei tentativi sono esportati nel JSON; in caso di parità viene mantenuto il primo migliore. Non si garantisce il minimo globale.

L'esperimento del gomito ripete questa procedura per **K da 1 a 10**. Per K=3 riutilizza il risultato già ottenuto con le stesse 20 inizializzazioni. Nel grafico si cerca il punto oltre il quale aumentare K riduce poco la SSE. Il gomito **non è sempre chiaramente identificabile** e dipende anche dalla standardizzazione. K=3 è inoltre una scelta motivata dalle tre specie conosciute, usate solo per progettare l'esperimento. Non è una prova che il dataset abbia necessariamente tre gruppi sferici separati. I minimi globali della SSE non crescono con K, mentre esecuzioni finite con inizializzazioni indipendenti possono occasionalmente non rispettare questa proprietà.

## Valutazione e risultati effettivi

La **silhouette** usa soltanto feature e cluster: confronta la distanza media interna `a` con la distanza media `b` verso il gruppo alternativo più vicino, tramite `(b-a)/max(a,b)`, quindi media sui campioni. Varia tra -1 e 1; valori maggiori indicano migliore separazione. È calcolata nello spazio standardizzato in quattro dimensioni. Con un solo cluster o un cluster per campione non è definita: il modulo restituisce `None`.

La **purezza** confronta cluster e specie. Le righe della tabella di contingenza sono le specie, le colonne i cluster:

\[
\text{purezza}=\frac{\sum_k \max_s n_{s,k}}{N}.
\]

Ogni cluster è valutato secondo la propria specie maggioritaria. La purezza non impone un abbinamento uno a uno tra specie e cluster e tende a favorire un numero elevato di cluster: da sola non basta per scegliere K.

L'**Adjusted Rand Index (ARI)** confronta le coppie di campioni nei due raggruppamenti correggendo l'accordo atteso per caso. Vale 1 per partizioni coincidenti, circa 0 per accordo casuale e può essere negativo. ARI e purezza non dipendono dalla numerazione dei cluster. La silhouette misura la geometria interna; purezza e ARI misurano l'accordo con una classificazione esterna.

<!-- RISULTATI_INIZIO -->
| Misura | Valore |
| --- | --- |
| Iterazioni del tentativo migliore | 4 |
| Convergenza raggiunta | Sì |
| Inerzia finale | 139.820496 |
| Silhouette | 0.459948 |
| Purezza | 83.33% |
| Adjusted Rand Index | 0.620135 |
| Campioni nei cluster 0, 1, 2 | 47, 50, 53 |
| Seed dell'inizializzazione selezionata | 275121930 |

Tabella di contingenza ottenuta:

| Specie | Cluster 0 | Cluster 1 | Cluster 2 |
| --- | --- | --- | --- |
| Iris setosa | 0 | 50 | 0 |
| Iris versicolor | 11 | 0 | 39 |
| Iris virginica | 36 | 0 | 14 |

Centroidi finali in centimetri (le colonne seguono l'ordine delle quattro feature):

| Cluster | Lunghezza sepalo | Larghezza sepalo | Lunghezza petalo | Larghezza petalo |
| --- | --- | --- | --- | --- |
| 0 | 6.7809 | 3.0957 | 5.5106 | 1.9723 |
| 1 | 5.0060 | 3.4280 | 1.4620 | 0.2460 |
| 2 | 5.8019 | 2.6736 | 4.3698 | 1.4132 |

Il cluster 1 contiene tutti e soli i 50 fiori setosa. Gli altri due cluster mescolano versicolor e virginica: la purezza corrisponde a (36 + 50 + 39) / 150 = 125 / 150. La silhouette positiva suggerisce una separazione geometrica parziale; l'ARI mostra che la partizione non coincide con quella botanica. I nomi delle specie restano soltanto un riferimento esterno.

Risultati del metodo del gomito:

| K | Inerzia minima su 20 tentativi |
| --- | --- |
| 1 | 600.000000 |
| 2 | 222.361705 |
| 3 | 139.820496 |
| 4 | 114.304803 |
| 5 | 90.927514 |
| 6 | 80.790198 |
| 7 | 71.855167 |
| 8 | 63.530739 |
| 9 | 54.525116 |
| 10 | 50.870636 |

Questi valori sono stati letti dagli output dell'esecuzione, non usati come costanti nel codice. I numeri dei cluster possono cambiare con seed o versioni differenti; confrontare partizioni e metriche, non il significato degli identificatori.
<!-- RISULTATI_FINE -->

## Grafici e loro interpretazione

Tutte le figure vengono salvate in `results/plots/`. **L'algoritmo usa sempre quattro feature standardizzate**: le figure sui petali sono proiezioni bidimensionali e non mostrano tutta la geometria utilizzata per assegnare i cluster. Nei grafici in centimetri i centroidi sono trasformati mediante `scaler.inverse_transform`.

| File | Lettura |
| --- | --- |
| `01_kmeans_clusters.png` | Lunghezza e larghezza dei petali, colori per cluster arbitrario e croci sui centroidi in cm. |
| `02_real_species.png` | Stesse coordinate con colori per specie reale; i colori dei cluster non implicano un'identità con quelli delle specie. |
| `03_elbow_method.png` | SSE minima dei 20 tentativi per ogni K; cercare una diminuzione dei benefici marginali. |
| `04_convergence.png` | SSE del tentativo selezionato a ogni aggiornamento, con stato iniziale all'indice 0. |
| `05_cluster_sizes.png` | Numero di fiori in ciascun cluster, che non deve necessariamente essere 50. |
| `06_contingency_matrix.png` | Conteggi delle combinazioni specie–cluster; i valori fuori dalla specie maggioritaria mostrano la mescolanza. |
| `07_centroid_movement.png` | Traiettorie dei centroidi proiettate sui petali; quadrato iniziale, croce finale e frecce del movimento. Uno spostamento in altre coordinate può essere invisibile nella proiezione. |

![Cluster individuati](results/plots/01_kmeans_clusters.png)
![Metodo del gomito](results/plots/03_elbow_method.png)
![Confronto con le specie](results/plots/06_contingency_matrix.png)

## Organizzazione dei file

```text
iris-kmeans/
├── README.md
├── requirements.txt
├── requirements-lock.txt          # Versioni esatte dell'ambiente verificato
├── .gitignore
├── main.py
├── src/
│   ├── __init__.py
│   ├── kmeans.py                  # Algoritmo NumPy e migliore di 20 tentativi
│   ├── data_loader.py             # Iris, CSV originali e StandardScaler
│   ├── evaluation.py              # Contingenza, purezza, silhouette e ARI
│   └── visualization.py           # Sette grafici, backend senza finestre
├── data/
│   ├── iris_features.csv
│   └── iris_labels.csv
├── results/
│   ├── plots/                     # Sette immagini PNG
│   ├── cluster_assignments.csv    # Feature originali, cluster e specie per confronto
│   ├── centroids.csv              # Centroidi in cm e nello spazio standardizzato
│   ├── metrics.json               # Metriche, parametri, versioni e inizializzazioni
│   ├── contingency_matrix.csv
│   ├── elbow_results.csv
│   ├── convergence.csv
│   └── centroids_history.csv      # Traiettorie in cm di tutti i centroidi
└── tests/
    └── test_kmeans.py
```

`requirements.txt` contiene intervalli di compatibilità per installazioni su Python diversi. `requirements-lock.txt` registra tutte le versioni installate nell'ambiente verificato; per riprodurre quello stesso ambiente su una piattaforma e versione Python compatibili: `python -m pip install -r requirements-lock.txt`. Gli output sono inclusi nella consegna e possono essere rigenerati. `.gitignore` esclude ambiente virtuale e cache, non i risultati didattici.


## Test e controlli

La suite usa `unittest`, incluso nella libreria standard. Verifica forma e valori finiti dei centroidi, etichette, riproducibilità, SSE non crescente su più seed, cronologie, `predict`, `fit_predict`, errori sugli input, K eccessivo, dati duplicati e identici, arresto dopo una sola iterazione o con tolleranza alta, media per K=1 e coerenza numerica finale. Controlla inoltre la scelta del minimo tra 20 tentativi e l'indipendenza delle metriche dalla numerazione dei cluster.

<!-- VERIFICA_INIZIO -->
Verifica effettuata con **Python 3.14.8** e le versioni registrate in `requirements-lock.txt` e `metrics.json`.

- Compilazione della sintassi di tutti i moduli: completata senza errori.
- **14 test unittest eseguiti, tutti superati** (inclusi controlli su 15 seed per la cronologia).
- Pipeline `main.py` eseguita fino al termine anche da una directory diversa da quella del progetto.
- Controllati dimensioni e corrispondenza dei CSV, K da 1 a 10, conteggi della contingenza, SSE ricalcolata dai CSV e conversione dei centroidi in centimetri.
- Verificata la presenza e la leggibilità dei sette PNG.

L'ambiente `.venv` è stato creato nel progetto e le dipendenze sono state installate. Su questa condivisione Windows il caricamento delle librerie binarie è risultato molto lento: le esecuzioni sulla condivisione sono state interrotte e la verifica completa è stata effettuata usando un ambiente temporaneo **locale con le stesse versioni**, eseguendo direttamente i sorgenti del workspace. Non restano verifiche funzionali sospese.

Se si riscontra lo stesso rallentamento, si può tenere il progetto sulla rete e creare l'ambiente sul disco locale. Dal terminale nella cartella del progetto:

```powershell
$irisVenv = Join-Path $env:LOCALAPPDATA 'venvs/iris-kmeans'
python -m venv $irisVenv
& (Join-Path $irisVenv 'Scripts/python.exe') -m pip install -r requirements.txt
& (Join-Path $irisVenv 'Scripts/python.exe') main.py
& (Join-Path $irisVenv 'Scripts/python.exe') -m unittest discover -s tests -v
```

In VS Code selezionare quel percorso locale come interprete Python. Gli output continuano a essere salvati nel progetto.
<!-- VERIFICA_FINE -->

## Limiti e conclusioni

K-means richiede K in anticipo, dipende dall'inizializzazione e dalla scala delle feature, risente degli outlier e rappresenta meglio gruppi compatti e approssimativamente sferici con dispersioni simili. Non scopre necessariamente le specie botaniche e non garantisce l'ottimo globale. Venti inizializzazioni riducono il rischio di un minimo locale sfavorevole senza eliminarlo. In dataset grandi il broadcasting può richiedere troppa memoria; qui si privilegiano chiarezza e leggibilità.

Il progetto permette di osservare il legame tra distanza, assegnazione, media e riduzione della SSE, e di distinguere la qualità geometrica dei cluster dall'accordo con etichette esterne. La valutazione combina quindi grafici, silhouette, purezza e ARI, mantenendo separati apprendimento e confronto con le specie reali.

