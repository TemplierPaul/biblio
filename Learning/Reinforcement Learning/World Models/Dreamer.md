# The Dreamer Series

A sequence of Model-Based Reinforcement Learning (MBRL) agents that learn a world model (learning physics from pixels) and then learn behaviors inside this analytic "imagination".



J'ai un début d'idée inspirée par Digital Red Queen et NeuPL:

- DRQ utilise un LLM pour générer du code Core War, mais l'eval se fait avec toutes les solutions précédentes qui se battent toutes ensemble, ça ne s'applique pas pour du 1v1. Pourtant à la fin leur eval est en 1v1 contre des programmes humains. 
- NeuPL a un seul ANN conditionné par un simplex qui indique qui est son opposant, et ça permet d'interpoler pour jouer contre une mixture d'opposants (et donc approximer le Nash). Mais avec du code, on a des programmes séparés, pas 1 seul ANN. 

Si on veut garder l'évolution de code, on peut avoir le LLM qui génère un code python, qui prend en entrée le vecteur de simplex qui définit l'opposant, et c'est le code python qui génère le code assembleur pour Core War.