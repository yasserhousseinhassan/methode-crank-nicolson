# La Méthode de Crank-Nicolson - Résolution de l'Équation de la Chaleur

Ce projet tutoré a été réalisé dans le cadre d'une **Licence de Mathématiques Fondamentales** à l'**Université de Djibouti** (Faculté des Sciences). Il est consacré à l'étude théorique et à l'implémentation numérique de la **méthode de Crank-Nicolson** pour résoudre des équations aux dérivées partielles (EDP) paraboliques.

## Informations Générales
* **Encadré par** : Dr. Souleiman Omar Hoch
* **Réalisés par** : Mikael Ketema Tarekegn, Mohamed Goumaneh Houssein, Yasser Houssein Hassan & Youssouf Khaireh Omar
* **Année universitaire** : 2021/2022

---

## Sommaire et Concepts Clés

### 1. Classification des EDP Linéaires du Second Ordre
* Définition générale et classification sous la forme $A u_{xx} + B u_{xy} + C u_{yy} + ... = G(x,y)$ :
  * **EDP Hyperbolique** ($B^2 - 4AC > 0$) : ex. Équation des ondes.
  * **EDP Parabolique** ($B^2 - 4AC = 0$) : ex. Équation de la chaleur/diffusion.
  * **EDP Elliptique** ($B^2 - 4AC < 0$) : ex. Équation de Laplace / Poisson.

### 2. Principes des Différences Finies (DF)
* Discrétisation spatio-temporelle du domaine sur un maillage régulier.
* Développement de Taylor pour approcher les dérivées partielles d'ordre 1 et 2.
* Analyse mathématique des schémas numériques :
  * **Erreur de troncature** et ordre du schéma.
  * **Consistance** (l'approximation tend vers l'équation continue quand les pas tendent vers 0).
  * **Stabilité** (norme $L^\infty$) : Schéma explicite (conditionnellement stable, $\Delta t \leq \frac{1}{2}\Delta x^2$) et schéma implicite (inconditionnellement stable).
  * **Convergence** (Lax équivalence théorème : Consistance + Stabilité $\implies$ Convergence).

### 3. Schéma de Crank-Nicolson
* Méthode de différences finies de type implicite introduite en 1947 par John Crank et Phyllis Nicolson.
* Conçue comme la moyenne arithmétique des schémas explicite et implicite.
* **Avantages** : Inconditionnellement stable en temps et en espace, précision d'ordre 2 en temps et en espace ($O(\Delta x^2 + \Delta t^2)$).
* **Limitations** : Possibilité d'oscillations numériques si le pas en temps est trop grand par rapport au pas en espace.

### 4. Application Numérique & Implémentation Matlab
Résolution de l'équation de la chaleur monodimensionnelle :
$$\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2}$$
* **Conditions aux limites** : Dirichlet homogènes ($u(0,t) = u(1,t) = 0$).
* **Condition initiale** : $u(x,0) = \sin(\pi x)$.
* **Solution analytique exacte** : $u(x,t) = e^{-\pi^2 t}\sin(\pi x)$.
* **Programmation Matlab** : Codes sources inclus dans le rapport pour :
  * Calcul de la solution analytique exacte.
  * Résolution par schéma explicite, implicite, et Crank-Nicolson (résolution de système tridiagonal).
  * Comparaison graphique et numérique démontrant la supériorité de Crank-Nicolson en termes de précision.

## Fichier du Projet
* `projet tutoré complet.pdf` : Le document complet du mémoire (36 pages) avec toutes les démonstrations mathématiques et les résultats de simulation 2D et 3D sous Matlab.

---
*Université de Djibouti*
