"""K-means implementato esclusivamente con NumPy."""

import numpy as np


def _positive_integer(value, name):
    """Rifiuta anche booleani e valori decimali apparentemente interi."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} deve essere un intero positivo.")


def _as_matrix(X):
    """Converte i dati e controlla forma, dimensioni e valori finiti."""
    try:
        matrix = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("X deve contenere valori numerici.") from error
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("X deve essere una matrice non vuota (campioni, feature).")
    if not np.isfinite(matrix).all():
        raise ValueError("X non deve contenere NaN o valori infiniti.")
    return matrix


class KMeansFromScratch:
    """Algoritmo di Lloyd con inizializzazione casuale di campioni distinti.

    La cronologia include lo stato iniziale (indice 0) e uno stato per ogni
    aggiornamento. Ogni inerzia usa i centroidi e le assegnazioni più vicine
    dello stesso stato. Per cluster vuoti si sceglie un campione lontano dai
    centroidi precedenti: si sostituisce solo un centroide senza membri.
    Con dati duplicati possono rimanere meno di K cluster occupati.
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-6, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _squared_distances(X, centers):
        """Broadcasting: (N, 1, D) - (1, K, D) produce (N, K, D)."""
        differences = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        return np.sum(differences ** 2, axis=2)

    def fit(self, X):
        """Apprende i centroidi dalle sole feature; non riceve etichette reali."""
        _positive_integer(self.n_clusters, "n_clusters")
        _positive_integer(self.max_iter, "max_iter")
        if not np.isscalar(self.tol) or not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol deve essere un numero finito non negativo.")
        X = _as_matrix(X)
        if self.n_clusters > len(X):
            raise ValueError("n_clusters non può superare il numero di campioni.")

        self.n_features_in_ = X.shape[1]

        """ Scelta degl indici per individuare i centroidi"""
        rng = np.random.default_rng(self.random_state) # Generatore a partire dal seed
        indices = rng.choice(len(X), size=self.n_clusters, replace=False) # scelta dei k indici


        """ Calcolo matrice delle distanze quadratiche e restituzione di una matrice con N righe e K colonne """
        distances = self._squared_distances(X, centers)

        """ Assegnazione al centroide più vicino """
        labels = np.argmin(distances, axis=1)
        #axis = 1 dice ad argmin di confrontare tra loro i valori contenuti nelle colonne presenti sulla singola riga della matrice.
        #axis = 0 dice ad argmin di confrontare tra loro i valori contenuti nelle righe presenti sulla singola colonna della matrice.
        
        """ Creo uno storage dei centroidi inziali """
        self.centroids_history_ = [centers.copy()]

        """ Calcolo l'inerzia (o SSE) """
        self.inertia_history_ = [float(distances[np.arange(len(X)), labels].sum())]

        """ Azzero il contatore dei riposizionamenti dei centroidi """
        self.empty_cluster_relocations_ = 0

        """ Imposto l'indicatore di convergenza a 0 """
        self.converged_ = False

        """ Ciclo di aggiornamento dei centroidi non vuoti"""
        for iteration in range(1, self.max_iter + 1):
            new_centers = centers.copy() #copia i centroidi attuali
            empty_clusters = [] # lista vuota per annotare gli indici dei cluster senza membri
            for cluster in range(self.n_clusters):
                members = X[labels == cluster] # seleziona le righe di X appartenente al cluster corrente
                if len(members):
                    new_centers[cluster] = members.mean(axis=0) # calcola il nuovo centroide facendo la media di ciascuna delle 4 feature
                else:
                    empty_clusters.append(cluster)

            # Il centroide vuoto non contribuisce alla SSE delle assegnazioni
            # correnti: sostituirlo non può peggiorare il minimo delle distanze.
            # Usiamo indici distinti per i recuperi nella stessa iterazione.
            
            #creo la matrice con le distanze di ciascun fiore con il centroide assegnato da labels
            residuals = distances[np.arange(len(X)), labels].copy()
            for cluster in empty_clusters:
                farthest = int(np.argmax(residuals)) #argmax restituisce l'indice di posizione del valore di distanza massima, tale indice coincide con l'indice dell'i-esimo fiore
                new_centers[cluster] = X[farthest] #assegna le coordinate delle feature del fiore farthest a new_center in posizione cluster
                residuals[farthest] = -np.inf #metto negativo il valore maggiore così non lo scelgo più
                self.empty_cluster_relocations_ += 1 #incremento il contatore che conta quante volte l'algoritmo è intervenuto per gestire un cluster vuoto durante l'addestramento

            """ Ricalcolo distanze, cluster di appartenenza, somma errori quadratici """
            new_distances = self._squared_distances(X, new_centers)
            new_labels = np.argmin(new_distances, axis=1)
            inertia = float(new_distances[np.arange(len(X)), new_labels].sum())

            """ Salviamo storia dei valori dell'SSE e dei centroidi """
            self.inertia_history_.append(inertia)
            self.centroids_history_.append(new_centers.copy())

            # Norma euclidea di tutti gli spostamenti (norma di Frobenius).
            shift = float(np.linalg.norm(new_centers - centers)) # calcolo il valore della dist. euclidea tra valori dei centri nuovi e vecchi
            unchanged = np.array_equal(new_labels, labels) # confronto gli array delle appartenenze ai cluster di ciasun fiore per vedere se sono stati fatti assegnamenti diversi rispetto il ciclo precedente 

            """ Aggiorniamo centri, appartenenze e matrice delle distanze """
            centers, labels, distances = new_centers, new_labels, new_distances

            if unchanged or shift <= self.tol: # se non ci sono assegnazioni a cluster diverse rispetto prima o lo spostamento dei centri è sotto un certo valore abbiamo convergenza e interrompiamo il ciclo
                self.converged_ = True
                break

        """ Salvo i valori trovati dei centri, delle appartenenze, dell'inerzia e il numero di iterazioni nell'oggetto """
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = iteration
        return self

    def predict(self, X):
        """Assegna nuove osservazioni, già nello stesso spazio delle feature."""
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Eseguire fit prima di predict.")
        X = _as_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Il numero di feature deve coincidere con quello di fit.")
        return np.argmin(self._squared_distances(X, self.cluster_centers_), axis=1)

    def fit_predict(self, X):
        """Addestra e restituisce una copia delle assegnazioni finali."""
        return self.fit(X).labels_.copy()


def fit_best_kmeans(X, n_clusters=3, n_init=20, random_state=42, max_iter=300, tol=1e-6):
    """Esegue n_init tentativi con seed distinti e sceglie la SSE minima.

    Non usa mai le specie reali. L'oggetto restituito conserva anche le inerzie
    di tutti i tentativi, utili per verificare la scelta del migliore.
    """
    _positive_integer(n_init, "n_init")
    rng = np.random.default_rng(random_state) # generatore random
    seeds = rng.choice(2**31 - 1, size=n_init, replace=False) # array di dim size che contiene i seed
    best = None
    inertias = []
    for seed in seeds:
        model = KMeansFromScratch(n_clusters, max_iter, tol, int(seed)).fit(X)
        inertias.append(model.inertia_)
        if best is None or model.inertia_ < best.inertia_:
            best = model
    best.initialization_inertias_ = inertias
    best.initialization_seeds_ = [int(seed) for seed in seeds]
    return best
