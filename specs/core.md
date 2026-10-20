# Spécification du core Microchat

## 1. Introduction

### Objectif

Le core fournit un chatbot interactif en terminal, fonctionnant localement sur CPU avec un modèle au format GGUF. Il prend en charge la configuration, la résolution et le chargement du modèle, les sessions de conversation, le rendu des réponses et l’intégration de plugins-agent indépendants.

### Contexte et périmètre

Le cœur du produit est réparti principalement entre `src/microchat/app.py`, `src/microchat/plugins.py` et `src/microchat/agents/runtime.py`. `src/microchat/__main__.py` et le script `microchat` lancent l’application. Les agents spécialisés sont distribués séparément et déclarent leurs plugins via le groupe d’entry points Python `microchat.plugins`.

Le core ne réalise pas lui-même de recherche métier externe. Il fournit au modèle les outils déclarés par les plugins actifs et délègue à ceux-ci leurs règles fonctionnelles. Le modèle et l’historique de conversation sont locaux ; les appels réseau éventuels sont propres aux plugins.

## 2. Spécifications fonctionnelles

### Démarrage et configuration

- L’application est démarrée depuis le répertoire contenant `config.ini`, avec `microchat` ou `python -m microchat`.
- Le paramètre `--version` affiche la version du paquet et termine le programme.
- Le fichier `config.ini` et la section `[microchat]` sont obligatoires. La configuration est lue sans interpolation INI.
- Tous les paramètres attendus dans `[microchat]` doivent être présents. Tout paramètre inconnu dans cette section ainsi que toute section autre que `[microchat]` ou `[plugin:<nom>]` provoque une erreur explicite.
- Les paramètres propres aux plugins sont transmis sous forme de chaînes à la fabrique du plugin. Leur interprétation et leur validation sont à la charge de celui-ci.
- La liste `plugins` est séparée par des virgules et les entrées vides sont ignorées. Les doublons sont refusés. Une liste vide active un chatbot sans outils spécialisés.
- Les noms configurés doivent correspondre à des entry points installés. Un plugin absent entraîne un message indiquant de l’installer ou de le retirer de la configuration.

### Réglages de l’application

| Groupe | Paramètres | Règles fonctionnelles |
|---|---|---|
| Modèle | `model_repo_id`, `model_path`, `models_dir` | `model_repo_id` ne peut pas être vide. Un `model_path` non vide désigne un fichier local existant et prend priorité sur le dépôt distant. `models_dir` désigne le cache des modèles téléchargés. |
| CPU et contexte | `n_threads`, `n_ctx`, `n_batch`, `use_mmap`, `use_mlock`, `flash_attn`, `warm_up` | `n_threads` est positif ou nul ; zéro demande une détection automatique. `n_ctx` est au moins égal à 512 et `n_batch` au moins égal à 1. Les options booléennes pilotent le chargement llama.cpp et le préchauffage. |
| Génération | `max_tokens`, `temperature`, `top_p`, `top_k`, `repeat_penalty`, `system_prompt` | `max_tokens` est positif. `temperature` appartient à `[0, 2]`, `top_p` à `]0, 1]`, `top_k` est positif ou nul et `repeat_penalty` est positif ou nul. Le prompt système est transmis au modèle avant les instructions des plugins. |
| Historique | `prompt_reserve_tokens`, `history_low_watermark` | Le budget d’historique est `n_ctx - max_tokens - prompt_reserve_tokens` et doit être positif. Le seuil de purge appartient à `]0, 1]`. |
| Session et rendu | `session_id`, `render_refresh_hz`, `render_interval_seconds`, `verbose` | L’identifiant détermine le fil de conversation. La fréquence Rich doit être au moins 1 et l’intervalle de rendu positif ou nul. `verbose` active les journaux détaillés. |

### Modèle et inférence

- En présence d’un fichier local configuré, le core vérifie son existence et le charge directement.
- Sinon, il réutilise un GGUF présent dans le cache du dépôt. Si le cache est vide, il consulte le dépôt Hugging Face, sélectionne un GGUF et le télécharge dans le cache.
- La sélection privilégie, dans cet ordre, les quantifications `Q4_K_M`, `Q4_K_S`, `Q4_0`, `Q5_K_M`, `Q5_K_S`, `Q6_K`, `Q8_0`, puis `F16`. À défaut, le premier nom de fichier GGUF trié est retenu. L’absence de GGUF est une erreur.
- L’inférence est configurée sans couche GPU. Si `n_threads = 0`, le nombre de cœurs physiques est privilégié, avec repli sur le nombre logique disponible.
- Si `warm_up` est activé, un prompt minimal est exécuté au démarrage. Un échec de préchauffage produit un avertissement et ne bloque pas l’application.
- Si le prompt système contient `/no_think`, le core désactive le mode de réflexion lorsque le template llama.cpp expose le paramètre correspondant.

### Conversation et interface

- Le core présente un prompt utilisateur, ignore les lignes vides et traite les commandes sans les envoyer au modèle.
- `/help`, `/h` et `/?` affichent l’aide ; `/clear` et `/reset` effacent la session ; `/info` affiche les informations de modèle, CPU, contexte, plugins actifs et nombre de messages ; `/quit`, `/exit` et `/q` quittent l’application.
- Une commande commençant par `/` mais inconnue est signalée. Les autres lignes deviennent des messages utilisateur.
- Ctrl+D ou la fin du flux termine la session. Ctrl+C au prompt est ignoré avec indication de la commande de sortie. Ctrl+C pendant une réponse interrompt cette génération et restaure l’état antérieur de la conversation.
- Les réponses et résultats d’outils sont rendus en Markdown avec Rich. Les icônes associées aux outils actifs sont affichées lors de leur exécution.
- À la fin d’une réponse, l’interface indique le nombre de tokens estimé, le débit, le délai avant le premier token et la durée totale. Les réponses issues de la recherche automatique affichent une durée de recherche dédiée.
- Les erreurs de génération sont journalisées et présentées dans le terminal sans arrêter la boucle interactive.

### Historique et cycle d’agent

- L’historique est isolé par `session_id` et stocké dans un checkpointer mémoire du processus. Il n’est pas persistant après l’arrêt de l’application.
- Avant chaque appel modèle, si le nombre estimé de tokens dépasse le budget, l’historique est réduit aux derniers messages complets compatibles avec le seuil `history_low_watermark`, en commençant par un message utilisateur.
- Les instructions non vides des plugins sont ajoutées au prompt système. Leurs outils et middleware sont réunis dans un agent LangGraph unique.
- Les noms d’outils doivent être uniques entre plugins actifs. Les icônes déclarées doivent référencer des outils appartenant au plugin.
- Les modèles locaux peuvent émettre des appels d’outils sous forme de balises textuelles Hermes/Qwen. Le runtime convertit les JSON valides en appels structurés, retire les blocs `<think>` et `<tool_call>` du texte visible et tolère les caractères parasites après le JSON.
- Le comptage de tokens utilise le tokenizer llama.cpp ; il sert à borner l’historique et à produire les métriques affichées.

## 3. Spécification technique

### Modules et dépendances

- `app.py` porte `AppConfig`, le chargement/validation INI, la résolution GGUF, la construction du modèle, le compteur de tokens et la boucle terminal `ChatApp`.
- `plugins.py` définit le contrat `AgentPlugin` et charge les fabriques à partir des entry points `microchat.plugins`.
- `agents/runtime.py` assemble l’agent LangGraph, les outils et middleware des plugins, purge l’historique et normalise les appels d’outils textuels des modèles locaux.
- `__main__.py` appelle `microchat.app.run`; le point d’entrée console déclaré par le paquet appelle la même fonction.
- Les principales bibliothèques sont LangChain, LangGraph, `llama-cpp-python`, Hugging Face Hub, Pydantic, Rich et psutil. La recherche web est une dépendance optionnelle du plugin `web_search`.

### Configuration et erreurs

- Le chemin par défaut de configuration est `Path.cwd() / "config.ini"`; il n’est pas recherché relativement au fichier du paquet.
- `load_config` convertit les types à l’aide de `configparser`, développe `~` dans les chemins locaux, et retourne une structure immuable `AppConfig`.
- Le budget d’historique est une propriété calculée de `AppConfig` ; une configuration laissant moins d’un token d’historique est rejetée.
- Le démarrage intercepte les erreurs de configuration, d’environnement fichier et de récupération du modèle prévues par `run`, les écrit sur stderr et retourne le code 1. Le code de retour normal est 0.
- La découverte utilise `importlib.metadata.entry_points(group="microchat.plugins")`. Chaque fabrique reçoit le mapping d’options de `[plugin:<nom>]` et doit retourner un `AgentPlugin` portant le nom attendu.

### Construction du modèle et du graphe

- `resolve_model_path` utilise un répertoire de cache dérivé de l’identifiant du dépôt, en remplaçant `/` par `--`. Les fichiers GGUF disponibles dans ce répertoire sont triés avant sélection.
- `build_llm` construit `ChatLlamaCpp` avec `n_gpu_layers=0`, les paramètres de contexte et de génération du fichier INI, et le nombre de threads également appliqué au prefill.
- `make_token_counter` met en cache les comptages de textes et ajoute une marge fixe par message et pour l’ensemble de la conversation.
- `build_agent` construit un agent avec `InMemorySaver`, les plugins actifs, un middleware `before_model` de réduction de l’historique, les middleware plugins, puis un middleware `after_model` de normalisation des appels textuels.
- Le graphe est identifié par `microchat_agent`. L’adaptateur de l’interface fournit le `session_id` comme `thread_id` LangGraph.

### Exécution et rendu

- `ChatApp` consomme `agent.stream(..., stream_mode="messages")` et distingue les fragments assistant des résultats `ToolMessage`.
- Le rendu est rafraîchi selon les paramètres Rich. Le contenu technique des appels d’outils n’est pas affiché comme une réponse normale.
- Avant une génération, l’état LangGraph existant est conservé. Si l’itération est interrompue ou échoue après modification de l’état, l’agent est remis à cet état précédent et le flux est fermé.
- Le nombre de messages affiché exclut les appels d’outils assistant et compte les messages utilisateur, les résultats outil et les réponses assistant finales.

### Contraintes d’exploitation

- Python 3.12 ou supérieur est requis. Le modèle est exécuté localement sur CPU, mais son premier téléchargement nécessite un accès au dépôt Hugging Face.
- La météo et la recherche web nécessitent un accès réseau lorsqu’elles sont utilisées. Une configuration sans plugins permet de converser avec le modèle hors ligne après téléchargement du GGUF.
- Le checkpointer étant en mémoire, aucune reprise de conversation entre deux exécutions n’est garantie.