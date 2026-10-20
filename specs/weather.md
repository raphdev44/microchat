# Spécification du plugin météo

## 1. Introduction

### Objectif

Le plugin `weather` fournit les conditions météorologiques actuelles et des prévisions quotidiennes ou par partie de journée. Il interprète les dates formulées en français, tente une localisation IP sans ville explicite et produit directement la restitution sans reformulation par le LLM.

### Contexte et périmètre

Le plugin est implémenté dans `src/microchat/agents/weather.py` et enregistré sous l’entry point `weather`. Il contribue l’outil `get_weather`, l’icône 🌦️, des instructions système et deux middleware. Il utilise les API publiques Open-Meteo de géocodage et de prévision, sans clé API.

Le plugin ne réalise pas de recherche générale et ne remplace pas les alertes météo officielles. Il dépend de la couverture Open-Meteo. La localisation IP via ipapi.co est approximative et peut correspondre à un VPN ou proxy plutôt qu'à la position réelle.

## 2. Spécifications fonctionnelles

### Routage et demande de ville

- L’outil météo n’est présenté au modèle que si le dernier message utilisateur correspond à une demande météo reconnue. En dehors de ces demandes, il est retiré de la liste d’outils disponible pour le modèle.
- Lors d’une demande météo reconnue, le plugin force le choix de `get_weather` lorsqu’il est disponible.
- Les termes météo reconnus incluent notamment météo, température, pluie, neige, orage, prévisions météo, « quel temps » et les questions « il va faire chaud » ou « fera-t-il froid » ; la détection est fondée sur une expression régulière et ne constitue pas une compréhension linguistique générale.
- Sans ville explicite, le tool tente une géolocalisation via `https://ipapi.co/json/`. Une ville inventée par le modèle est supprimée de l'appel. Une ville explicitement fournie évite la géolocalisation IP.
- Le fournisseur de géolocalisation voit l'adresse IP publique de la machine. La localisation estimée est signalée dans la restitution ; aucune clé API n'est nécessaire.
- Si la géolocalisation échoue ou renvoie des coordonnées invalides, le tool demande « Pour quelle ville voulez-vous la météo ? ». La date et le nombre de jours sont conservés dans l'artefact de résultat. La réponse suivante est interprétée comme une ville, après retrait d'un préfixe « à », « au », « aux » ou « pour ». Une nouvelle question météo est traitée comme telle.
- Les noms de ville sont comparés sans tenir compte des majuscules, accents ou ponctuation pour vérifier qu’ils figurent bien dans le message utilisateur.

### Entrées de l’outil

`get_weather` accepte les paramètres suivants :

| Paramètre | Règles |
|---|---|
| `city` | Facultatif ; absent ou `null` pour tenter la géolocalisation IP. Sinon, nom non vide après suppression des espaces périphériques, utilisé pour le géocodage. |
| `when` | Facultatif ; expression française transmise telle que formulée. Aujourd’hui, du jour, demain, après-demain, « dans N jours », jour de semaine, jour du mois, date jour/mois, date ISO, mois en lettres, cette semaine ou partie de journée. Par défaut, le jour local courant. |
| `days` | Nombre de jours consécutifs à retourner à partir de `when`, entre 1 et 16 inclus. Valeur par défaut : 1. |

- Pour un jour du mois sans mois ni année, le prochain jour valide à partir d’aujourd’hui est retenu, éventuellement dans le mois suivant. Un jour et un mois sans année donnent la prochaine occurrence correspondante.
- Un jour de semaine désigne la prochaine occurrence ; si c’est le jour courant, la date courante est utilisée.
- Les dates passées sont refusées. Une date située après la période couverte par l’API est refusée avec la dernière date disponible indiquée.
- Si une demande de plusieurs jours dépasse la fin de la période couverte, seuls les jours disponibles sont renvoyés et une note le signale.
- « Cette semaine » couvre aujourd'hui jusqu'au dimanche inclus, selon la date locale retournée par Open-Meteo.
- Les créneaux horaires sont matin (6 h–12 h), après-midi (12 h–18 h), soir (18 h–24 h) et nuit (0 h–6 h du jour sélectionné). Les expressions telles que « demain après-midi » conservent leur date et leur créneau.

### Données retournées

- Le géocodage sélectionne le premier résultat Open-Meteo. Le nom affiché inclut, lorsque disponibles, la ville, la région administrative et le pays.
- Pour aujourd’hui, la réponse inclut les conditions courantes : code et libellé météo, température, ressenti, humidité relative, précipitations et vitesse du vent.
- Chaque jour demandé comprend le libellé météo, les températures minimale et maximale, le cumul de précipitations, la probabilité maximale de précipitation et le vent maximal.
- Pour une partie de journée, seuls les horaires de ce créneau sont agrégés : températures minimale et maximale, cumul de pluie, probabilité et vent maximaux. Le libellé correspond à l'heure la plus pluvieuse disponible. Les conditions « maintenant » ne sont pas ajoutées. Si les prévisions horaires sont absentes, une erreur explicite remplace tout cumul journalier trompeur.
- Les codes WMO connus sont traduits en libellés français et en symboles. Un code inconnu est rendu comme « conditions inconnues » ; une valeur manquante est rendue `n/d`.
- Le tool génère un texte Markdown et un artefact JSON sérialisable `microchat_panel` contenant le type de rendu, le titre et le corps. L'interface affiche une carte Rich pour un jour ou un créneau et un panneau pour plusieurs jours. Le texte reste disponible pour l'historique et les consommateurs sans rendu Rich.
- Ces résultats constituent directement la réponse finale : le modèle ne reformule pas les données météo et n'est pas rappelé après le tool.

### Erreurs et limites de service

- Une ville sans résultat de géocodage donne une erreur explicite « Aucune ville trouvée ».
- Une réponse de prévision absente ou invalide est signalée comme indisponibilité ou réponse invalide du service.
- Une erreur HTTP 429 ou 5xx, une erreur URL ou un timeout fait l’objet d’un seul nouvel essai après une attente de 250 ms. Les autres erreurs HTTP sont remontées sans nouvel essai.
- Le délai réseau est configurable. La requête météo échoue si le service ne répond pas dans ce délai ou après le retry autorisé.

## 3. Spécification technique

### Contrat et configuration

- `load_plugin(config)` retourne un `AgentPlugin` nommé `weather`.
- L’option `timeout_seconds`, issue de `[plugin:weather]`, est un entier supérieur ou égal à 1 ; sa valeur par défaut est 30 secondes.
- Le plugin expose l’outil `get_weather`, l’icône associée `🌦️` et les instructions qui imposent l'utilisation du tool, l'omission de la ville sans localisation explicite et le passage à l'identique de la période.
- La fabrique historique `build_agent` reste disponible comme adaptateur de compatibilité ; elle compose le runtime commun avec le seul plugin météo.

### Appels Open-Meteo

- Le géocodage utilise `https://geocoding-api.open-meteo.com/v1/search` avec le nom, un résultat, la langue française et le format JSON.
- La prévision utilise `https://api.open-meteo.com/v1/forecast`, avec latitude et longitude géocodées ou estimées par IP, données `current` et `daily`, `forecast_days=16` et `timezone=auto`. Les données `hourly` sont demandées uniquement pour une partie de journée.
- Les requêtes sont formées avec `urllib.request.Request` et `urlencode`, l’en-tête `Accept: application/json` et l’agent utilisateur `microchat/0.1`.
- `get_json` valide que la réponse décodée est un objet JSON. Les séries journalières sont indexées à partir de la date locale fournie par la première entrée `daily.time`.
- Le résultat est un couple texte Markdown et artefact. L'outil LangChain est configuré avec `response_format="content_and_artifact"`, `return_direct=True` et `handle_tool_error=True`. L'artefact contient uniquement des données sérialisables pour rester compatible avec les checkpoints LangGraph.

### Dates et formatage

- `resolve_day` normalise espaces et apostrophes, traite les expressions relatives, accepte les dates ISO et extrait les jours/mois/années partiels avec des expressions régulières.
- Les noms des jours et des mois sont définis dans le module afin de ne pas dépendre de la locale système. Le premier jour d’un mois est rendu sous la forme « 1er ».
- `normalize_name` réalise une comparaison insensible aux accents en normalisant Unicode, puis en supprimant ponctuation et séparateurs.
- Le tableau `_WEATHER_CODES` associe les codes WMO supportés à leur symbole et leur libellé français.

### Middleware

- Le middleware `wrap_model_call` force l’outil météo sur une demande détectée ou lorsqu’une ville est attendue après la question de précision ; sinon il masque l’outil au modèle.
- Le middleware `after_model` vérifie qu'une ville appelée figure dans le dernier message utilisateur ; sinon il l'efface et laisse le tool tenter la localisation IP.
- Après un échec de localisation, il retrouve `pending_weather` dans l'artefact du dernier `ToolMessage`. À la réponse suivante, il crée l'appel structuré `get_weather` à partir de la ville et des paramètres en attente. L'identifiant du message assistant courant est préservé lors de son remplacement.

### Tests de référence

Les tests vérifient les dates françaises, la semaine calendaire, les agrégats horaires limités au créneau demandé, les cartes et panneaux sérialisables, leur affichage à 40 et 100 colonnes, la limite de 16 jours, les retries, la géolocalisation et son repli avec conservation de date, les appels Qwen/Hermes, l'absence de reformulation LLM et le masquage de l'outil hors contexte météo.