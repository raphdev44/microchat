# Microchat

Microchat est un chatbot interactif en terminal capable de fonctionner sur un simple ordinateur portable sans GPU.
Il s'agit d'une expérimentation d'IA Frugale pour construire un chatbot crédible pouvant mobiliser des agents spécialisés et des outils.

Microchat s'appuie sur des modèles GGUF en local avec
`llama.cpp` et permet d'ajouter ou de retirer des plugins-agent indépendamment du cœur.

| Élément | Technologie |
|---|---|
| Inférence | `llama.cpp` via `llama-cpp-python`, CPU uniquement |
| Modèles | GGUF locaux ou téléchargés depuis Hugging Face |
| Agent et outils | LangChain et LangGraph |
| Interface | Rich : Markdown, indicateur d'activité et métriques de génération |
| Configuration | `config.ini` |

## Installation

Python 3.12 ou supérieur est requis. Sous Windows, utilisez Python 3.12 64 bits. Depuis la
racine du projet, créez l'environnement et installez le projet :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu "llama-cpp-python>=0.3"
python -m pip install -e ".[dev]"
```

L'index CPU officiel fournit une roue précompilée de `llama-cpp-python`. Sans roue compatible,
l'installation depuis PyPI nécessite un compilateur C/C++ et CMake pour compiler `llama.cpp`
avec les optimisations disponibles sur la machine (AVX2, AVX-512, NEON, etc.).

## Démarrage

Le fichier `config.ini` est requis dans le répertoire courant. Lancez Microchat depuis le
dossier qui le contient :

```powershell
microchat
# ou : python -m microchat
```

Au premier lancement, le modèle GGUF est téléchargé dans `~/.cache/microchat/models`. Les
lancements suivants utilisent le cache local. Le modèle fonctionne hors ligne après le
téléchargement; les plugins qui consultent des services Internet nécessitent une connexion.

Les commandes disponibles sont `/help`, `/clear`, `/info` et `/quit` (ou Ctrl+D). Ctrl+C
pendant une réponse interrompt la génération sans quitter; le tour interrompu n'est pas
mémorisé. `/info` affiche notamment les plugins actifs.

## Configuration

Tous les réglages se trouvent dans `config.ini`. Pour changer de modèle, modifiez
`model_repo_id`. L'application découvre les fichiers GGUF du dépôt et privilégie `Q4_K_M`
quand elle est disponible; le cache est séparé par dépôt. Définissez `model_path` pour utiliser
un fichier GGUF déjà présent sur la machine.

Les paramètres du modèle, du contexte, de la génération, de l'historique et de l'affichage
sont regroupés dans `[microchat]`. Les options propres aux plugins sont documentées dans la
section suivante et placées dans leur section `[plugin:<nom>]`.

## Plugins

Les plugins apportent des outils spécialisés au chatbot. Microchat découvre les plugins
installés via le groupe d'entry points Python `microchat.plugins`, puis n'active que les noms
listés dans `plugins` sous `[microchat]`. Dans la configuration fournie, la météo est activée
par défaut et la recherche web est disponible en option.

Pour désactiver un plugin, retirez son nom de la liste. Pour n'activer aucun plugin, laissez
la valeur vide :

```ini
[microchat]
plugins =
```

Les réglages d'un plugin résident dans `[plugin:<nom>]`. Un plugin nommé dans `plugins` mais
non installé provoque une erreur explicite au démarrage.

### Météo (`weather`)

Le plugin météo utilise l'API publique Open-Meteo, sans clé API. Il fournit les conditions
actuelles et des prévisions jusqu'à 16 jours pour une ville. Les dates en français comme
« demain », « samedi » ou « le 21 » sont interprétées selon la date locale de la ville; les
prévisions peuvent couvrir plusieurs jours. Une ville est demandée si elle manque.

La consultation nécessite Internet. Configurez son délai réseau dans `config.ini` :

```ini
[plugin:weather]
timeout_seconds = 30
```

### Recherche web (`web_search`)

Le plugin recherche avec DuckDuckGo les informations récentes ou celles dont le modèle n'est
pas sûr. Il renvoie des titres, des extraits et leurs URL; il ne télécharge pas le contenu
complet des pages. Le modèle reçoit la consigne de citer les URL, de signaler les incertitudes
et de traiter les extraits web comme des données non fiables.

Installez la dépendance optionnelle `ddgs` :

```powershell
python -m pip install -e ".[search]"
```

Ajoutez ensuite le plugin à la liste existante dans `[microchat]` et configurez-le :

```ini
[microchat]
plugins = weather, web_search

[plugin:web_search]
region = wt-wt
timeout_seconds = 5
max_results = 5
```

`max_results` doit être compris entre 1 et 10. Toute recherche nécessite Internet. Pour
désactiver seulement la recherche web, retirez `web_search` de la liste et conservez les
autres plugins actifs.

### Ajouter un plugin

Un plugin déclare un entry point dans le paquet Python qui le fournit :

```toml
[project.entry-points."microchat.plugins"]
research = "microchat_research:load_plugin"
```

La fonction `load_plugin(config)` reçoit les options de `[plugin:research]` et retourne un
`AgentPlugin` importé depuis `microchat.plugins`. Le plugin déclare ses instructions, ses
outils, les icônes associées et, si nécessaire, son middleware. Installez le paquet, puis
ajoutez `research` à `plugins`. Les noms des outils doivent être uniques parmi les plugins
actifs.

## Architecture

```text
src/microchat/
├── __main__.py        Point d'entrée du programme
├── app.py             Configuration, modèle, interface et sessions
├── plugins.py         Contrat et découverte des plugins
└── agents/
    ├── runtime.py     Assemblage commun du chatbot et des plugins actifs
    ├── weather.py     Plugin météo
    └── web_search.py  Plugin DuckDuckGo
```

Le runtime commun gère l'historique et les appels d'outils des modèles locaux. Chaque plugin
garde sa logique métier et sa configuration séparées du cœur.

## Réglages CPU

| Réglage | Emplacement | Effet |
|---|---|---|
| `n_threads = 0` | `config.ini` | Détecte les cœurs CPU physiques; cette valeur est aussi utilisée pour le prefill |
| `n_ctx`, `n_batch`, quantification | `config.ini` | Ajustent mémoire, contexte et débit selon le modèle |
| `warm_up` | `config.ini` | Prépare le modèle au démarrage pour réduire la latence du premier tour |
| Budget et purge de l'historique | `config.ini`, `agents/runtime.py` | Bornent le contexte et réduisent le recalcul sur CPU |
| Comptage des tokens | `app.py` | Utilise le tokenizer natif de `llama.cpp` avec cache |

## Développement

Installez les dépendances de développement avec `python -m pip install -e ".[dev]"`, puis
lancez les contrôles depuis la racine du dépôt :

```powershell
pytest
ruff check src tests
ruff format --check src tests
mypy src
```

Les tests n'ont pas besoin de télécharger un modèle GGUF ni de compiler `llama.cpp`.
