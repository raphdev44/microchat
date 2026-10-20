# Spécification du plugin météo

## 1. Introduction

### Objectif

Le plugin `weather` fournit les conditions météorologiques actuelles et des prévisions quotidiennes pour une ville. Il interprète les dates formulées en français, demande une ville si elle manque et renvoie directement le résultat Open-Meteo formaté à l’utilisateur.

### Contexte et périmètre

Le plugin est implémenté dans `src/microchat/agents/weather.py` et enregistré sous l’entry point `weather`. Il contribue l’outil `get_weather`, l’icône 🌦️, des instructions système et deux middleware. Il utilise les API publiques Open-Meteo de géocodage et de prévision, sans clé API.

Le plugin fournit uniquement une prévision à l’échelle d’une ville et par jour. Il ne réalise pas de recherche générale, ne remplace pas les alertes météo officielles et dépend de la disponibilité et de la couverture des services Open-Meteo.

## 2. Spécifications fonctionnelles

### Routage et demande de ville

- L’outil météo n’est présenté au modèle que si le dernier message utilisateur correspond à une demande météo reconnue. En dehors de ces demandes, il est retiré de la liste d’outils disponible pour le modèle.
- Lors d’une demande météo reconnue, le plugin force le choix de `get_weather` lorsqu’il est disponible.
- Les termes météo reconnus incluent notamment météo, température, pluie, neige, orage, prévisions météo et formulations de type « quel temps » ; la détection est fondée sur une expression régulière et ne constitue pas une compréhension linguistique générale.
- Une ville doit être explicitement nommée par l’utilisateur. Si le modèle propose une ville absente du message, le plugin n’appelle pas le service et demande « Pour quelle ville voulez-vous la météo ? ».
- La date et le nombre de jours demandés sont conservés pendant cette demande de précision. La réponse suivante de l’utilisateur est interprétée comme le nom de ville, après retrait d’un éventuel préfixe « à », « au », « aux » ou « pour ».
- Les noms de ville sont comparés sans tenir compte des majuscules, accents ou ponctuation pour vérifier qu’ils figurent bien dans le message utilisateur.

### Entrées de l’outil

`get_weather` accepte les paramètres suivants :

| Paramètre | Règles |
|---|---|
| `city` | Obligatoire, au moins un caractère et non composé uniquement d’espaces après suppression des espaces périphériques. Sert au géocodage. |
| `when` | Facultatif ; expression française transmise telle que formulée. Valeurs comprises : aujourd’hui, demain, après-demain, « dans N jours », jour de semaine, jour du mois, date jour/mois, date ISO ou date avec mois en lettres. Par défaut, le jour local courant de la ville. |
| `days` | Nombre de jours consécutifs à retourner à partir de `when`, entre 1 et 16 inclus. Valeur par défaut : 1. |

- Pour un jour du mois sans mois ni année, le prochain jour valide à partir d’aujourd’hui est retenu, éventuellement dans le mois suivant. Un jour et un mois sans année donnent la prochaine occurrence correspondante.
- Un jour de semaine désigne la prochaine occurrence ; si c’est le jour courant, la date courante est utilisée.
- Les dates passées sont refusées. Une date située après la période couverte par l’API est refusée avec la dernière date disponible indiquée.
- Si une demande de plusieurs jours dépasse la fin de la période couverte, seuls les jours disponibles sont renvoyés et une note le signale.

### Données retournées

- Le géocodage sélectionne le premier résultat Open-Meteo. Le nom affiché inclut, lorsque disponibles, la ville, la région administrative et le pays.
- Pour aujourd’hui, la réponse inclut les conditions courantes : code et libellé météo, température, ressenti, humidité relative, précipitations et vitesse du vent.
- Chaque jour demandé comprend le libellé météo, les températures minimale et maximale, le cumul de précipitations, la probabilité maximale de précipitation et le vent maximal.
- Les codes WMO connus sont traduits en libellés français et en symboles. Un code inconnu est rendu comme « conditions inconnues » ; une valeur manquante est rendue `n/d`.
- Les résultats sont fournis en Markdown, et constituent directement la réponse finale : le modèle ne reformule pas les données météo.

### Erreurs et limites de service

- Une ville sans résultat de géocodage donne une erreur explicite « Aucune ville trouvée ».
- Une réponse de prévision absente ou invalide est signalée comme indisponibilité ou réponse invalide du service.
- Une erreur HTTP 429 ou 5xx, une erreur URL ou un timeout fait l’objet d’un seul nouvel essai après une attente de 250 ms. Les autres erreurs HTTP sont remontées sans nouvel essai.
- Le délai réseau est configurable. La requête météo échoue si le service ne répond pas dans ce délai ou après le retry autorisé.

## 3. Spécification technique

### Contrat et configuration

- `load_plugin(config)` retourne un `AgentPlugin` nommé `weather`.
- L’option `timeout_seconds`, issue de `[plugin:weather]`, est un entier supérieur ou égal à 1 ; sa valeur par défaut est 30 secondes.
- Le plugin expose l’outil `get_weather`, l’icône associée `🌦️` et les instructions qui imposent l’utilisation de l’outil en cas de demande météo, la demande explicite de ville et le passage à l’identique de l’expression de date.
- La fabrique historique `build_agent` reste disponible comme adaptateur de compatibilité ; elle compose le runtime commun avec le seul plugin météo.

### Appels Open-Meteo

- Le géocodage utilise `https://geocoding-api.open-meteo.com/v1/search` avec le nom, un résultat, la langue française et le format JSON.
- La prévision utilise `https://api.open-meteo.com/v1/forecast`, avec latitude et longitude géocodées, données `current` et `daily`, `forecast_days=16` et `timezone=auto`.
- Les requêtes sont formées avec `urllib.request.Request` et `urlencode`, l’en-tête `Accept: application/json` et l’agent utilisateur `microchat/0.1`.
- `get_json` valide que la réponse décodée est un objet JSON. Les séries journalières sont indexées à partir de la date locale fournie par la première entrée `daily.time`.
- Le résultat est une chaîne Markdown. L’outil LangChain est configuré en `return_direct=True` et avec `handle_tool_error=True`.

### Dates et formatage

- `resolve_day` normalise espaces et apostrophes, traite les expressions relatives, accepte les dates ISO et extrait les jours/mois/années partiels avec des expressions régulières.
- Les noms des jours et des mois sont définis dans le module afin de ne pas dépendre de la locale système. Le premier jour d’un mois est rendu sous la forme « 1er ».
- `normalize_name` réalise une comparaison insensible aux accents en normalisant Unicode, puis en supprimant ponctuation et séparateurs.
- Le tableau `_WEATHER_CODES` associe les codes WMO supportés à leur symbole et leur libellé français.

### Middleware

- Le middleware `wrap_model_call` force l’outil météo sur une demande détectée ou lorsqu’une ville est attendue après la question de précision ; sinon il masque l’outil au modèle.
- Le middleware `after_model` vérifie qu’une ville appelée figure dans le dernier message utilisateur. Il sauvegarde `when` et `days` dans `additional_kwargs.pending_weather` lorsqu’il doit demander la ville.
- À la réponse suivante, il crée l’appel structuré `get_weather` à partir du texte utilisateur et des paramètres météo en attente. Le même identifiant de message assistant est réutilisé pour remplacer la question de précision lors de l’ajout à l’état LangGraph.

### Tests de référence

Les tests du plugin vérifient le formatage des données, les dates françaises, les prévisions multi-jours, la limite de 16 jours, les retries, la demande de ville, l’extraction des appels d’outils textuels Qwen/Hermes et le masquage de l’outil hors contexte météo.