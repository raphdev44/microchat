# microchat

Micro chatbot **terminal**, **100 % CPU**, construit autour de **LangChain** :

| Rôle | Choix |
|---|---|
| Moteur d'inférence | `llama.cpp` via `llama-cpp-python` |
| Modèle | Dépôt GGUF configurable dans `config.ini` |
| Orchestration | LangChain `create_agent` et LangGraph |
| Interface | `rich` (Markdown streamé, spinner, métriques tok/s) |
| Configuration | `config.ini` |

## Installation

Lance ces commandes depuis la racine du projet, dans le dossier qui contient `pyproject.toml`.

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu "llama-cpp-python>=0.3"
python -m pip install -e ".[dev]"
```

Le projet requiert Python 3.12 ou supérieur. Sous Windows, utilisez Python 3.12 64 bits.
L'index CPU officiel fournit une roue précompilée de `llama-cpp-python` pour éviter de
compiler `llama.cpp`. Sans roue binaire compatible, l'installation depuis PyPI nécessite
un compilateur C/C++ et CMake, et compile `llama.cpp` avec les optimisations de *votre* CPU
(AVX2, AVX-512, NEON…).

## Utilisation

```bash
microchat                      # ou : python -m microchat
```

Au premier lancement, le GGUF est téléchargé dans `~/.cache/microchat/models`. Ensuite le
modèle local fonctionne hors ligne ; seule la consultation météo requiert Internet.
Tous les réglages, y compris le modèle, le prompt, les performances et l'historique, sont
dans `config.ini`. Pour changer de modèle, modifiez uniquement `model_repo_id`. L'application
découvre les fichiers GGUF du dépôt et privilégie `Q4_K_M` quand elle est disponible ; le
cache est séparé par dépôt.

Commandes : `/help`, `/clear`, `/info`, `/quit` (ou Ctrl+D). Ctrl+C pendant une réponse
l'interrompt sans quitter ; ce tour n'est alors pas mémorisé.

Pour les questions météo, l'agent appelle l'outil Open-Meteo et répond pour une ville :
conditions actuelles, un jour précis (« demain à Montréal », « samedi », « le 21 »,
« 21/10 ») ou des prévisions sur plusieurs jours (« sur 3 jours », « cette semaine »),
jusqu'à 16 jours. Les dates sont interprétées selon la date locale de la ville, et les
réponses sont illustrées de symboles météo (☀️ ⛅ 🌧️ ⛈️ ❄️ 🌡️ 💨…). Cette API
publique ne demande pas de clé, mais une connexion Internet est nécessaire pour cette
fonctionnalité. L'historique par session est borné au budget de contexte configuré.

Le fichier `config.ini` est requis et lu depuis le répertoire courant.

## Structure

```
src/microchat/
├── __main__.py        Lance l'application
├── app.py             Configuration, modèle, téléchargement, mémoire et interface
└── agents/
    └── weather.py     Outil Open-Meteo, prompt et graphe LangChain/LangGraph
```

Pour ajouter un agent, crée un fichier dans `agents/` et garde son prompt, ses outils et
son graphe LangChain/LangGraph dans ce fichier. Déclare aussi son icône (`ICON`) et les noms
de ses outils (`TOOL_NAMES`), puis ajoute le module à `_AGENTS` dans `agents/__init__.py` :
l'icône s'affiche au-dessus de chaque réponse qui a mobilisé l'agent (🌦️ pour la météo).
Les valeurs de réglage restent dans `config.ini`.

## Réglages CPU

| Optimisation | Où | Effet |
|---|---|---|
| Threads = cœurs **physiques**, y compris `n_threads_batch` | `app.py` | Débit de génération et de prefill |
| Quantification et `use_mmap` | `config.ini` | Dépendent du dépôt sélectionné |
| **Warm-up** pendant le spinner | `app.py` | Pas de latence « disque » à la 1re question |
| Historique purgé **par paliers** | `agents/weather.py` | Réduit le recalcul du prompt sur CPU |
| Prompt système immuable, placé en tête | `config.ini` | Idem (préfixe en cache) |
| Comptage de tokens natif + cache | `app.py` | Pas de tokenizer GPT-2/`transformers` |
| Rendu Markdown limité | `config.ini` | Le CPU reste au modèle |

Pour régler les performances, modifiez `n_threads`, `n_batch`, `n_ctx` ou les paramètres
d'échantillonnage directement dans `config.ini`.

## Développement

```bash
pytest                 # Tests sans téléchargement de modèle ni compilation de llama.cpp
ruff check src && ruff format src
mypy src               # strict
```
