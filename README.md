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

```bash
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

`llama-cpp-python` se compile à l'installation (compilateur C/C++ + CMake requis) avec
les optimisations de *votre* CPU (AVX2, AVX-512, NEON…), ce qui est idéal. Pour éviter la
compilation, utilisez les roues précompilées CPU de la documentation du projet
(`--extra-index-url`), au prix d'un jeu d'instructions plus générique.

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

Pour les questions sur la météo actuelle, l'agent appelle l'outil Open-Meteo et peut
répondre pour une ville (par exemple « Quel temps fait-il à Montréal ? »). Cette API
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
son graphe LangChain/LangGraph dans ce fichier. Les valeurs de réglage restent dans
`config.ini`.

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
