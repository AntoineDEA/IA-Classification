import numpy as np 

## Variable entrée qui correspond à mon tableau de valeurs 

x_entrer = np.array(([3,1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1],[4.5,1]), dtype=float)

## Données de sortie avec 1 = rouge et 2 = Bleu

y = np.array(([1],[0],[1],[0],[1],[0],[1],[0]), dtype=float)

## Tous mettre entre 1 et 0 en divisant pas la valeur la plus grande.

x_entrer = x_entrer/np.amax(x_entrer, axis=0)

## Récupérer les 8 premières valeures ( La neuvième étant celle à prédire)

X = np.split(x_entrer,[8])[0]
xPrediction = np.split(x_entrer,[8])[1]

class Neural_Network(object):

    ## Initialisation 

    def __init__(self):
        ## Nombre de synapse d'entrée
        self.inputSize = 2 
        ## Nombre de synapse de sortie
        self.outputSize = 1
        ## Nombre de synapse cachés
        self.hiddenSize = 3
        ## Générer aléatoirement des poids entre synapse d'entrée et caché
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) ## Matrice 2*3
        ## Générer aléatoirement des poids entre synapse caché et de sortie
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) ## Matrice 3*1
    
    ## Fonction de propagation avant ( Multiplication + sigmoid )

    def forward(self,X):
        ## Produit matriciel entre nos valeurs de base et notre matrice de poids générer aléatoirement
        self.z = np.dot(X,self.W1)
        ## Application de la sigmoid à notre matrice nouvellement obtenue, on obtient donc nos valeurs cachées
        self.z2 = self.sigmoid(self.z)
        ## Produit matriciel entre nos valeurs cachées et notre matrice de poids générer aléatoirement
        self.z3 = np.dot(self.z2,self.W2)
        ## Application de la sigmoid à notre matrice nouvellement obtenue, on obtient donc nos valeurs finales
        o = self.sigmoid(self.z3)
        return o

    ## Fonction Sigmoid

    def sigmoid(self,s):
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self,s):
        return s * (1-s)

    ## Fonction de retour en arrière 

    def backward(self, X, y, o):
        self.o_error = y - o ## Calcul de l'erreur
        self.o_delta = self.o_error * self.sigmoidPrime(o) ## Calcul de l'erreur delta de ce neuronne de sortie 
        self.z2_error = self.o_delta.dot(self.W2.T) ## Erreur des 3 neuronnes cachés
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2) ## Erreur delta des trois neuronnes cachés
        ## Mise à jour des poids
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    ## Fonction d'entrainement de notre programme

    def train(self,X,y):
        o = self.forward(X)
        self.backward(X,y,o)

    ## Fonction qui vas permettre de prédire un résultat

    def predict(self):
        print("Donnéee prédite après entrainement: ")
        print("Entrée : \n" + str(xPrediction))
        print("Sortie : \n" + str(self.forward(xPrediction)))

        ## En fonction de la prédiction, précisé si la fleur est Bleu ou Rouge

        if(self.forward(xPrediction) < 0.5):
            print("La fleur est Bleu ! \n")
        else:
            print("La fleur est Rouge ! \n")


## Création de notre réseau de neuronnes

NN = Neural_Network()

for i in range(30000): ## J'entraine mon modèle 30000 fois afin d'affiner les résultats
    print("#" + str(i) + "\n")
    print("Valeurs d'entrées : \n" + str(X))
    print("Sortie actuelle : \n" + str(y))
    print("Sortie prédite  : \n" + str(np.matrix.round(NN.forward(X),2))) ## On affiche seulement deux décimales pour éviter des nombre à virgule infinis
    print("\n")
    NN.train(X,y) ## Call de la fonction d'entrainement
    
NN.predict() ## Call de la fonction de prédiction












