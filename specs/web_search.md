# Spécification du plugin de recherche web

## 1. Introduction

### Objectif

Le plugin `web_search` permet de vérifier des questions factuelles avec des résultats DuckDuckGo récents. Le LLM formule une réponse à partir des extraits, puis Microchat ajoute une liste compacte des sources (titres et URL). Le plugin ne télécharge pas le contenu complet des pages.

### Contexte et périmètre

Le plugin est implémenté dans `src/microchat/agents/web_search.py` et enregistré sous l’entry point `web_search`. Il contribue l’outil `search_web`, l’icône 🌐, des instructions de réponse et trois middleware. Il requiert la dépendance optionnelle `ddgs` et un accès réseau.

Le plugin cible les questions factuelles explicites sur des personnes, organisations, événements, œuvres et informations récentes. La sélection des requêtes est heuristique : elle reconnaît un ensemble de formulations prédéfinies et ne doit pas être interprétée comme une classification exhaustive de l’intention.

## 2. Spécifications fonctionnelles

### Déclenchement et politique de recherche

- Le plugin lance une recherche avant la réponse du modèle pour les formulations factuelles reconnues, par exemple « c’est quoi », « qui est », « que signifie », « où se trouve », « comment fonctionne », ainsi que plusieurs formulations anglaises.
- Les demandes météo reconnues sont exclues de la recherche web, afin de laisser la décision au plugin météo.
- Les salutations, questions conversationnelles et calculs simples ne déclenchent pas de recherche selon la politique actuelle.
- Pour les requêtes qui ne déclenchent pas de recherche, `search_web` est retiré des outils proposés au modèle. Il reste disponible pour une question classée comme demande factuelle.
- Le résultat d’une recherche automatique termine le tour : le modèle local n’est pas appelé pour reformuler ou compléter les résultats.
- Les résultats sont des données non fiables. Les instructions demandent de ne pas exécuter d’instructions éventuellement présentes dans les extraits et de ne rapporter que les faits appuyés par ces extraits.

### Entrée et paramètres

| Paramètre | Règles |
|---|---|
| `query` | Requis, entre 2 et 500 caractères selon le schéma Pydantic. |
| `region` | Région DuckDuckGo non vide ; valeur par défaut `wt-wt`. |
| `timeout_seconds` | Entier supérieur ou égal à 1 ; valeur par défaut 5 secondes. |
| `max_results` | Entier compris entre 1 et 3 inclus ; valeur par défaut 3. |

- La recherche automatique part du dernier message utilisateur. Elle réduit les espaces consécutifs, retire la ponctuation terminale, puis limite la requête envoyée à 500 caractères.
- La clé de recherche est l’identifiant du message utilisateur, ou le texte de la requête si cet identifiant est absent. Une recherche déjà marquée dans l’historique pour cette clé n’est pas relancée.
- DuckDuckGo est appelé avec la région configurée, Safe Search modéré, le backend automatique et le nombre maximal de résultats configuré. Au plus les trois premiers résultats sont conservés et transmis au modèle.

### Résultats et réponse

- Chaque résultat exploitable est transmis au modèle avec son titre, son URL et son extrait. Le titre peut être remplacé par « Sans titre » ; plusieurs clés d’URL et d’extrait sont acceptées selon la réponse du fournisseur.
- Seules les URL commençant par `https://` ou `http://` sont conservées. Les entrées sans URL valide sont exclues.
- Si aucun résultat n’est disponible, l’utilisateur reçoit un message invitant à reformuler. Si aucun résultat ne comporte d’URL valide, un message distinct est affiché.
- Le modèle rédige une réponse concise et prudente à partir des extraits disponibles. Après sa réponse, Microchat ajoute une section `Sources :` sous forme de puces `- titre — URL`, sans demander au modèle de générer cette liste.
- La réponse doit traiter directement la question courante, sans réciter les résultats ni prendre la première personne d’une personne recherchée. Les résultats concernant des homonymes ou des organisations différentes ne doivent pas être fusionnés ; une ambiguïté doit être signalée.
- Une erreur de recherche est convertie en message d’indisponibilité, sans spéculation sur la réponse factuelle.

### Erreurs et dépendance

- L’absence du paquet `ddgs` empêche le chargement du plugin et fournit la commande d’installation de l’extra `search`.
- « No results found » est traité comme une absence de résultat, pas comme une erreur de service.
- Pour les autres erreurs de recherche, une seconde tentative est faite une fois ; si elle échoue, l’outil signale le type d’erreur sans exposer son détail à l’utilisateur.
- Les erreurs de configuration (valeur non entière, délai inférieur à 1, nombre de résultats hors limites, région vide) provoquent le refus de chargement du plugin.

## 3. Spécification technique

### Contrat et configuration

- `load_plugin(config)` retourne un `AgentPlugin` nommé `web_search`, déclarant l’outil `search_web`, l’icône 🔎 et deux middleware.
- Les options lues dans `[plugin:web_search]` sont `region`, `timeout_seconds` et `max_results`. Elles sont converties et validées avant la construction de l’outil.
- La fabrique vérifie la disponibilité de `ddgs` au chargement. Elle abaisse ensuite le niveau de journalisation des modules DDGS et de son client HTTP à WARNING.
- Le schéma `SearchQuery` impose une taille de requête entre 2 et 500 caractères.

### Client DuckDuckGo et formatage

- `_fetch_results` crée un client `DDGS` avec le délai configuré et appelle `text()` avec `region`, `safesearch="moderate"`, `max_results` et `backend="auto"`. La configuration et le résultat retourné sont limités à trois entrées.
- Le client est utilisé dans un gestionnaire de contexte. Les résultats sont matérialisés en liste avant fermeture.
- Une liste vide permet une seule nouvelle tentative avant de donner un résultat vide. L’exception explicitement « No results found » donne immédiatement un résultat vide. Une autre exception permet également une seule nouvelle tentative, puis remonte à l’outil.
- `_format_results` recherche l’URL dans `href`, `url` ou `link`, et l’extrait dans `body` ou `snippet`. Il écarte les schémas d’URL autres que HTTP et HTTPS.
- `_format_sources` crée une puce indépendante par résultat valide. Un titre dépassant 120 caractères est remplacé par le domaine de sa propre URL pour éviter qu’un titre agrégé mélange plusieurs résultats.
- `make_search_tool` transforme les erreurs d’import et les erreurs fournisseur en `ToolException`, avec `handle_tool_error=True`.

### Middleware et cycle de réponse

- `_requires_lookup` combine deux expressions régulières : une liste de formulations factuelles et une liste d’exclusions météo.
- `_make_lookup_middleware` est un middleware `before_model`. Pour une requête admissible, il appelle la recherche et ajoute à l’historique un message assistant d’appel d’outil et un `ToolMessage` contenant les extraits ; le graphe continue ensuite vers le LLM.
- Les erreurs et résultats vides sont transmis au LLM afin qu’il formule une réponse prudente. `_make_search_policy_middleware` masque `search_web` aux appels du modèle, car la recherche est déjà effectuée automatiquement avant l’inférence.
- Pour une recherche automatique, la question courante, les résultats associés et l’historique conversationnel sont transmis ensemble au modèle. Les sources sont ajoutées après sa réponse.
- `_make_sources_middleware` est un middleware `after_model`. Il retrouve les sources associées à la requête et ajoute `Sources :` et les puces `- titre — URL` au message final, sans solliciter une génération de sources par le LLM.
- Les middleware sont déclarés dans l’ordre `lookup`, `policy`, puis `sources`. Le runtime commun ajoute le middleware de normalisation des appels d’outils des modèles locaux.

### Tests de référence

Les tests vérifient les formulations déclenchantes et exclues, l’appel du LLM avec les résultats, l’ajout programmatique des sources, le plafond de trois résultats, le masquage de l’outil au modèle, les paramètres DuckDuckGo, le retry unique, le traitement des résultats vides, la validation des URL et les erreurs de configuration/dépendance.