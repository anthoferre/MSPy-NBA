# MSPy-NBA

Projet fil-rouge lors de ma formation Data-Scientist chez Data-Scientest

Objectifs principaux :
- Comparer les tirs (fréquence et efficacité au tir par situation de jeu et par localisation sur le terrain) de 20 des meilleurs joueurs de NBA du 21ème siècle (selon ESPN) : 
    - Tim Duncan
    - Kobe Bryant
    - Allen Iverson
    - Steve Nash
    - Ray Allen
    - Paul Pierce
    - Pau Gasol
    - Tony Parker
    - Manu Ginobili
    - Dwayne Wade
    - LeBron James
    - Chris Paul
    - Kevin Durant
    - Russell Westbrook
    - Stephen Curry
    - James Harden
    - Kawhi Leonard
    - Damian Lillard
    - Anthony Davis
    - Giannis Antetokounmpo

- Pour chacun de ces 20 joueurs encore actifs aujourd’hui (de LeBron James à Giannis Antetokounmpo), estimer à l’aide d’un modèle la probabilité qu’a leur tir de rentrer dans le panier, en fonction de métriques telles que : 

La phase de la compétition (saison régulière ou playoffs).
Le lieu du match (domicile ou extérieur).
Le nom de l’équipe adverse.
La conférence de l’équipe adverse (le niveau de la conférence ouest est souvent considéré plus élevé que celui de la conférence est).
Le classement de l’équipe adverse à la fin de la saison régulière.
La distance entre le joueur et le panier au moment du tir.
L’angle entre le joueur et la ligne de fond au moment du tir.
Le type de tir tenté (dunk, …)
Le type d’action menant au tir (pick and role, …)
Le temps restant avant la fin du quart temps au moment du tir.
Le score au moment du tir (mène ou mené).
Le numéro du quart-temps au cours duquel est pris le tir (1er, 2ème, 3ème ou 4ème quart-temps).
Le temps restant avant la fin du quart-temps.
etc …


Nous avions 4 jeux de données sur les saisons de 2000 à 2020, des données personelles et d'équipes, généralisées à une saison, un match ou par action de match que l'on peut retrouver sur les liens kaggle ci-dessous :
- https://www.kaggle.com/jonathangmwl/nba-shot-locations
- https://sports-statistics.com/sports-data/nba-basketball-datasets-csv-files/
- https://www.kaggle.com/nathanlauga/nba-games?select=ranking.csv
- https://www.kaggle.com/drgilermo/nba-players-stats?select=Players.csv
