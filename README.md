# FakeNewsMLModel
/!\ Ces instructions sont à destination d'un système Windows. Les commandes peuvent varier selon le système d'exploitation. /!\

Comment exécuter le projet : 
1 - Sommaire
2 - Détails de l'exécution

1 - Sommaire :
a) Pré-requis
b) Installation d'un environnement virtuel Python
c) Installation des dépendances
d) Exécution des migrations
e) Lancement du site web


2 - Détail de l'exécution 

a) Pré-requis

Ce projet est codé en Python 3.9, vous devez donc avoir Python 3.9 d'installé sur votre système pour pouvoir lancer le site web.
Clonez le projet depuis Github : 

```
cd path/to/directory
git clone https://github.com/romainrbn/FakeNewsMLModel.git
```

b) Installation d'un environnement virtuel Python

Une fois Python installé, placez-vous dans un dossier où vous voulez installer votre environnement virtuel.

```
cd C:/Users/myuser/environments
```

Une fois ceci fait, créez l'environnement virtuel et activez-le :

```
python3 -m venv /path/to/new/virtual/environment
cd /path/to/new/virtual/environment/Scripts/
activate.bat
```

c) Installation des dépendances

Pour rendre l'installation user-friendly, nous avons créé un fichier de dépendances lisible par PIP.
Placez-vous à la racine du projet et lancez pip avec le fichier requirements.txt :

```
cd path/to/project
pip install -r requirements.txt
``` 

d) Exécution des migrations
