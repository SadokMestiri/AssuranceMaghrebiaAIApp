#!/usr/bin/env python
"""
Train the intent classifier and save to models/intent_classifier.pkl.

Pipeline: normalize → TF-IDF (word 1-2grams + char 3-5grams) → LinearSVC with
Platt-scaling calibration → calibrated probabilities.

Usage:
    python train_intent_classifier.py              # synthetic dataset only
    python train_intent_classifier.py --with-eval  # merge real eval_store data
    python train_intent_classifier.py --eval-only  # real data only (needs ≥10/class)
    python train_intent_classifier.py --dry-run    # print CV report, don't save
"""
from __future__ import annotations

import argparse
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Any

# ── Dataset (synthetic, 25 examples × 13 intents = 325 total) ─────────────────
_RAW: list[tuple[str, str]] = [
    # ── explain ──────────────────────────────────────────────────────────────
    ("Pourquoi ce client a un score impayé si élevé ?", "explain"),
    ("Quels facteurs influencent le risque de résiliation ?", "explain"),
    ("Explique les facteurs de risque SHAP pour ce contrat", "explain"),
    ("Pourquoi le modèle prédit une fraude pour cette police ?", "explain"),
    ("Quels features sont les plus importants pour le churn ?", "explain"),
    ("Explique la décision du modèle sur ce client", "explain"),
    ("Pourquoi ce sinistre est classé à risque élevé ?", "explain"),
    ("Quels éléments ont conduit à ce score de risque ?", "explain"),
    ("Donne-moi les variables SHAP pour la police 12345", "explain"),
    ("Comment le modèle évalue-t-il le risque d'impayé ?", "explain"),
    ("Pourquoi ce client est catégorisé comme risque élevé ?", "explain"),
    ("Explique l'importance des variables pour ce modèle", "explain"),
    ("Quels facteurs contribuent le plus à la résiliation AUTO ?", "explain"),
    ("Pourquoi la prime de ce contrat est si élevée ?", "explain"),
    ("Quel est le poids de chaque variable dans la prédiction ?", "explain"),
    ("Explique la contribution des features au score impayé", "explain"),
    ("Comment est calculé le score de fraude pour ce sinistre ?", "explain"),
    ("Quels sont les drivers du taux de résiliation IRDS ?", "explain"),
    ("Pourquoi ce client risque de ne pas payer ?", "explain"),
    ("Interprète le modèle de segmentation pour ce profil", "explain"),
    ("Donne une explication SHAP pour les anomalies détectées", "explain"),
    ("Comment le modèle identifie-t-il les fraudes potentielles ?", "explain"),
    ("Quels critères influencent le bonus-malus de ce client ?", "explain"),
    ("Explique pourquoi ce contrat est signalé comme atypique", "explain"),
    ("Facteurs de risque pour les impayes en branche SANTE", "explain"),

    # ── predict ───────────────────────────────────────────────────────────────
    ("Lance le modèle de prédiction de fraude sur cette police", "predict"),
    ("Prédit le risque d'impayé pour ce client", "predict"),
    ("Fais tourner le modèle de résiliation sur le portefeuille AUTO", "predict"),
    ("Applique le modèle ML sur ce lot de contrats", "predict"),
    ("Score de risque via gradient boosting pour cette police", "predict"),
    ("Modèle de prédiction : risque de churn pour IRDS 2024", "predict"),
    ("Random forest : prédit la sévérité sinistre pour ce dossier", "predict"),
    ("Modèle ML : probabilité d'impayé pour la police 9876", "predict"),
    ("Lance le scoring fraude sur les sinistres du mois", "predict"),
    ("Prédiction machine learning du risque de résiliation SANTE", "predict"),
    ("Modélisation du risque pour les nouveaux contrats", "predict"),
    ("Scoring ML : identifie les polices à risque élevé", "predict"),
    ("Fais une prédiction gradient boosting sur ce portefeuille", "predict"),
    ("Modèle de détection de fraude : résultats sur les sinistres Q4", "predict"),
    ("Lancer le modèle de prédiction d'impayé sur branche AUTO", "predict"),
    ("Applique random forest pour scorer les nouveaux assurés", "predict"),
    ("Gradient boosting prediction résiliation : résultats 2024", "predict"),
    ("Utilise le modèle ML pour prédire les impayes Q1 2025", "predict"),
    ("Scoring machine learning : polices à surveiller en IRDS", "predict"),
    ("Prédit via ML la probabilité de fraude pour ce sinistre", "predict"),
    ("Score de risque ML pour le segment premium", "predict"),
    ("Fais tourner le modèle gradient boosting de fraude", "predict"),
    ("ML predict : taux de churn prévu pour le portefeuille global", "predict"),
    ("Applique le random forest scoring sur ce client VIP", "predict"),
    ("Modèle de prédiction : risque crédit assurance pour ce profil", "predict"),

    # ── dimension ─────────────────────────────────────────────────────────────
    ("Quels sont les clients par tranche d'âge ?", "dimension"),
    ("Distribution des polices par type de véhicule", "dimension"),
    ("Combien de clients femmes vs hommes en AUTO ?", "dimension"),
    ("Liste des produits par famille de risque", "dimension"),
    ("Répartition des agents par localité", "dimension"),
    ("Quels genres de véhicules sont couverts ?", "dimension"),
    ("Profil des assurés : personnes physiques vs morales", "dimension"),
    ("Top 10 marques de véhicules dans le portefeuille", "dimension"),
    ("Distribution des sinistres par nature", "dimension"),
    ("Nombre de polices par état (en cours, expirées)", "dimension"),
    ("Âge moyen des clients par branche", "dimension"),
    ("Catalogue de produits branche IRDS", "dimension"),
    ("Distribution par puissance véhicule", "dimension"),
    ("Nationalité des assurés dans le portefeuille", "dimension"),
    ("Nombre de familles de risque en SANTE", "dimension"),
    ("Liste des sinistres par responsabilité", "dimension"),
    ("Top villes des clients assurés", "dimension"),
    ("Répartition des polices par périodicité de paiement", "dimension"),
    ("Profil des véhicules assurés : marque, genre, puissance", "dimension"),
    ("Nombre de produits distincts en branche AUTO", "dimension"),
    ("Distribution des contrats par état (E, P, A)", "dimension"),
    ("Quelles sont les familles de risque disponibles ?", "dimension"),
    ("Clients par groupe d'agent commercial", "dimension"),
    ("Distribution bonus-malus dans le portefeuille", "dimension"),
    ("Sinistres : répartition par état de dossier", "dimension"),

    # ── overview ──────────────────────────────────────────────────────────────
    ("Donne-moi une vue globale de la situation", "overview"),
    ("Synthèse complète des KPI et risques 2024", "overview"),
    ("État global du portefeuille : tout ce que tu sais", "overview"),
    ("Tableau de bord complet de la performance", "overview"),
    ("Diagnostic complet de l'activité Maghrebia", "overview"),
    ("Toutes les informations disponibles sur AUTO", "overview"),
    ("Vue d'ensemble : production, risques, anomalies, alertes", "overview"),
    ("Synthèse décisionnelle du trimestre", "overview"),
    ("Résumé global de toutes les métriques clés", "overview"),
    ("Situation globale : KPI + drift + anomalies + segments", "overview"),
    ("Rapport complet toutes branches confondues", "overview"),
    ("Montre-moi tout ce qui est disponible sur IRDS", "overview"),
    ("Vue consolidée : production + sinistres + impayes + forecast", "overview"),
    ("Synthèse exécutive pour le comité de direction", "overview"),
    ("État global des opérations Maghrebia Q4 2024", "overview"),
    ("Toutes les infos : production, risque, drift, segments", "overview"),
    ("Rapport de situation global pour toutes les branches", "overview"),
    ("Briefing complet : métriques, alertes, tendances", "overview"),
    ("Que se passe-t-il en ce moment ? Vue complète.", "overview"),
    ("Synthèse intégrale : finance + risque + commercial", "overview"),
    ("Dashboard complet avec tous les indicateurs disponibles", "overview"),
    ("Vue 360° de la performance Maghrebia", "overview"),
    ("Donne-moi toutes les informations stratégiques", "overview"),
    ("Diagnostic global : anomalies, drift, KPI, prévisions", "overview"),
    ("Situation complète de l'entreprise pour une prise de décision", "overview"),

    # ── forecast ──────────────────────────────────────────────────────────────
    ("Prévision de la prime nette sur 3 mois", "forecast"),
    ("Projette le taux de résiliation pour les 6 prochains mois", "forecast"),
    ("Forecast des sinistres branche AUTO Q1 2025", "forecast"),
    ("Quelle sera la production IRDS le mois prochain ?", "forecast"),
    ("Anticipe l'évolution des impayes sur 4 mois", "forecast"),
    ("Prédis le chiffre d'affaires pour 2025", "forecast"),
    ("Projection du ratio S/P sur l'horizon 3 mois", "forecast"),
    ("Prévision de la prime acquise SANTE pour 2025", "forecast"),
    ("Forecast impayes : combien le mois prochain ?", "forecast"),
    ("Quelle sera l'évolution du taux de résiliation ?", "forecast"),
    ("Projette les sinistres sur les 3 prochains mois", "forecast"),
    ("Prévision de la production globale horizon 6 mois", "forecast"),
    ("Anticipe les tendances de prime nette IRDS", "forecast"),
    ("Quel est le forecast de résiliation pour Q2 2025 ?", "forecast"),
    ("Prévois l'évolution du ratio combiné sur 3 mois", "forecast"),
    ("Projection future des KPI principaux", "forecast"),
    ("Quelle sera la tendance des primes AUTO en 2025 ?", "forecast"),
    ("Prévision mensuelle du coût sinistres sur 6 mois", "forecast"),
    ("Forecast du nombre de sinistres pour les prochains mois", "forecast"),
    ("Projette le taux d'impayé horizon 3 mois", "forecast"),
    ("Prédis la prime nette de la branche SANTE pour Q1 2025", "forecast"),
    ("Quelle est la projection du chiffre d'affaires 2025 ?", "forecast"),
    ("Forecast 6 mois : production, impayes, résiliations", "forecast"),
    ("Anticipe l'évolution de la prime nette toutes branches", "forecast"),
    ("Prévision du ratio S/P pour l'année prochaine", "forecast"),

    # ── anomaly ───────────────────────────────────────────────────────────────
    ("Y a-t-il des anomalies contractuelles en AUTO ?", "anomaly"),
    ("Détecte les contrats atypiques dans le portefeuille IRDS", "anomaly"),
    ("Quels contrats présentent des comportements inhabituels ?", "anomaly"),
    ("Identifie les outliers dans les données de sinistres", "anomaly"),
    ("Y a-t-il des pics inhabituels dans les primes ?", "anomaly"),
    ("Signale les contrats suspects toutes branches", "anomaly"),
    ("Anomalies détectées : lesquelles sont critiques ?", "anomaly"),
    ("Détection de fraude : contrats à surveiller en SANTE", "anomaly"),
    ("Y a-t-il des ruptures dans les tendances de sinistres ?", "anomaly"),
    ("Quels assurés ont un comportement de paiement atypique ?", "anomaly"),
    ("Détecte les polices anormales avec score élevé", "anomaly"),
    ("Anomalies contractuelles : top 10 polices suspectes", "anomaly"),
    ("Identifie les contrats avec un ratio sinistres/prime anormal", "anomaly"),
    ("Y a-t-il des outliers dans les données d'impayes ?", "anomaly"),
    ("Détecte les comportements frauduleux potentiels", "anomaly"),
    ("Contrats atypiques SANTE : score d'anomalie élevé", "anomaly"),
    ("Quels sinistres présentent des caractéristiques inhabituelles ?", "anomaly"),
    ("Anomalies sur les polices AUTO 2024 : rapport complet", "anomaly"),
    ("Identifie les points aberrants dans le portefeuille", "anomaly"),
    ("Détecte les clients avec des patterns d'impayes anormaux", "anomaly"),
    ("Y a-t-il des incohérences dans les données sinistres ?", "anomaly"),
    ("Anomalies IRDS : quels contrats signaler à la direction ?", "anomaly"),
    ("Outliers détectés par Isolation Forest dans le portefeuille", "anomaly"),
    ("Quels contrats semblent frauduleux selon le modèle ?", "anomaly"),
    ("Détecte les ruptures structurelles dans la production mensuelle", "anomaly"),

    # ── drift ─────────────────────────────────────────────────────────────────
    ("Y a-t-il un drift statistique sur les données de prime ?", "drift"),
    ("La distribution des primes a-t-elle changé récemment ?", "drift"),
    ("Détecte une dérive dans les distributions AUTO", "drift"),
    ("Stabilité des données : a-t-on observé un drift ?", "drift"),
    ("Analyse la dérive statistique sur les 6 derniers mois", "drift"),
    ("Y a-t-il une dégradation de la qualité des données ?", "drift"),
    ("Drift PSI sur les features du modèle impayé", "drift"),
    ("Distribution des commissions : drift détecté ?", "drift"),
    ("La distribution du bonus-malus a-t-elle drifté ?", "drift"),
    ("Drift sur les données IRDS : résultats Evidently", "drift"),
    ("Analyse la stabilité des distributions de primes nettes", "drift"),
    ("Y a-t-il une dérive de population sur le portefeuille AUTO ?", "drift"),
    ("Dérive statistique : KS test sur les features de sinistres", "drift"),
    ("Drift détecté sur les données d'impayes en 2024 ?", "drift"),
    ("Stabilité du portefeuille : tests statistiques de drift", "drift"),
    ("Analyse la dérive des distributions SANTE vs période de référence", "drift"),
    ("PSI score : stabilité des features sur 12 mois", "drift"),
    ("Y a-t-il un changement de distribution significatif ?", "drift"),
    ("Drift sur les primes acquises : analyse courant vs référence", "drift"),
    ("Dégradation de la représentativité des données de modèle", "drift"),
    ("Drift KS/Chi² sur les variables catégorielles BRANCHE", "drift"),
    ("Y a-t-il une dérive saisonnière dans les données ?", "drift"),
    ("Dérive structurelle : 12 mois réf vs 6 mois courant", "drift"),
    ("Analyse la stabilité des features pour le modèle churn", "drift"),
    ("Distribution drift : rapport complet toutes features", "drift"),

    # ── segmentation ──────────────────────────────────────────────────────────
    ("Quels sont les segments clients et leur profil ?", "segmentation"),
    ("Segmentation RFM du portefeuille : résultats", "segmentation"),
    ("Donne-moi les clusters clients avec leur comportement", "segmentation"),
    ("Profil des clients VIP vs clients à risque", "segmentation"),
    ("Segmentation K-Means : combien de groupes distincts ?", "segmentation"),
    ("Quels sont les personas clients en AUTO ?", "segmentation"),
    ("Analyse la valeur vie client par segment", "segmentation"),
    ("Cluster des clients dormants : combien et qui sont-ils ?", "segmentation"),
    ("Segmentation IRDS : clients fidèles vs clients à risque", "segmentation"),
    ("Profil du segment avec le plus fort CLV", "segmentation"),
    ("Groupes de clients identifiés par machine learning", "segmentation"),
    ("Segmentation portefeuille : distribution et actions recommandées", "segmentation"),
    ("Quels clients appartiennent au segment Risque ?", "segmentation"),
    ("RFM clustering : résultats et interprétation", "segmentation"),
    ("Segmentation clients SANTE : qui sont les plus rentables ?", "segmentation"),
    ("Combien de clusters dans le portefeuille toutes branches ?", "segmentation"),
    ("Profil moyen de chaque segment client", "segmentation"),
    ("Segmentation par valeur : VIP, Fidèles, Potentiels", "segmentation"),
    ("K-Means sur les données clients : résultats 2024", "segmentation"),
    ("Analyse RFM : récence, fréquence, montant par segment", "segmentation"),
    ("Clients dormants : stratégie de réactivation recommandée ?", "segmentation"),
    ("Segmentation comportementale du portefeuille assurance", "segmentation"),
    ("Quels segments présentent le plus de risque d'attrition ?", "segmentation"),
    ("Profil des assurés par segment de valeur", "segmentation"),
    ("Distribution des clients dans les 5 segments identifiés", "segmentation"),

    # ── client ────────────────────────────────────────────────────────────────
    ("Qui est le client Jean Martin ?", "client"),
    ("Recherche le client Mehdi Bouazizi dans le portefeuille", "client"),
    ("Donne-moi les infos du client avec police 78901", "client"),
    ("Profil du client Ahmed Ben Salah", "client"),
    ("Y a-t-il des homonymes pour le client Fatma Trabelsi ?", "client"),
    ("Informations sur l'assuré avec CIN 08765432", "client"),
    ("Top clients par prime nette", "client"),
    ("Qui sont les meilleurs clients en terme de prime ?", "client"),
    ("Recherche l'assuré Mohamed Ali", "client"),
    ("Donne le profil complet du client VIP 12345", "client"),
    ("Historique des sinistres du client Karim Mansour", "client"),
    ("Identité et polices du client Nadia Dridi", "client"),
    ("Top 10 clients par montant d'impayes", "client"),
    ("Infos client : polices actives, sinistres de Sofiane Ben Ali", "client"),
    ("Recherche client par nom : Hedi Gharbi", "client"),
    ("Y a-t-il plusieurs assurés avec le nom Ben Slimane ?", "client"),
    ("Profil de risque du client avec référence 45678", "client"),
    ("Qui est l'assuré derrière la police AUTO-2024-9876 ?", "client"),
    ("Top 5 clients les plus profitables", "client"),
    ("Historique complet du client Rania Jebali", "client"),
    ("Trouve le client Yassine Zouari et ses contrats", "client"),
    ("Fiche client : contrats actifs et score de risque", "client"),
    ("Client VIP : détails polices et sinistralité", "client"),
    ("Recherche homonymes pour Sonia Mejri", "client"),
    ("Profil complet de l'assuré avec matricule 33456", "client"),

    # ── alerte ────────────────────────────────────────────────────────────────
    ("Y a-t-il des alertes critiques en cours ?", "alerte"),
    ("Quelles alertes sont actives en ce moment ?", "alerte"),
    ("Seuils dépassés : quelles métriques sont en alerte ?", "alerte"),
    ("Alertes impayes : taux dépassé en AUTO ?", "alerte"),
    ("Incidents de production détectés ce mois ?", "alerte"),
    ("Monitoring : y a-t-il des indicateurs en zone rouge ?", "alerte"),
    ("Surveillance : quels KPI ont déclenché une alerte ?", "alerte"),
    ("Alertes IRDS : taux impayé au-dessus du seuil ?", "alerte"),
    ("Liste des alertes par niveau de sévérité", "alerte"),
    ("Y a-t-il une alerte sur la production SANTE ?", "alerte"),
    ("Tableau de bord alertes : statut temps réel", "alerte"),
    ("Incidents actifs : résumé et actions recommandées", "alerte"),
    ("Alertes critiques : quelles décisions prendre ?", "alerte"),
    ("Surveillance impayes : seuil 2% dépassé quelque part ?", "alerte"),
    ("Alertes de production : baisse de 15% détectée ?", "alerte"),
    ("Quels indicateurs sont au-dessus des seuils de tolérance ?", "alerte"),
    ("Alertes actives branche AUTO : rapport de surveillance", "alerte"),
    ("Monitoring en temps réel : statut des alertes", "alerte"),
    ("Y a-t-il un incident sur le taux de résiliation ?", "alerte"),
    ("Alertes de performance : tous les seuils dépassés", "alerte"),
    ("Quelles alertes escalader à la direction ?", "alerte"),
    ("Surveillance ratio sinistres/prime : alerte déclenchée ?", "alerte"),
    ("Incident taux impayé IRDS : quelle est la situation ?", "alerte"),
    ("Alertes du jour : production, impayes, résiliations", "alerte"),
    ("Quelles métriques nécessitent une intervention immédiate ?", "alerte"),

    # ── rag ───────────────────────────────────────────────────────────────────
    ("Comment est calculé le FGA ?", "rag"),
    ("Qu'est-ce que la CNAM en assurance ?", "rag"),
    ("Quel est le rôle de la CGA ?", "rag"),
    ("Comment fonctionne le bonus-malus en AUTO ?", "rag"),
    ("Qu'est-ce que le ratio combiné ?", "rag"),
    ("Définition du PSAP en comptabilité d'assurance", "rag"),
    ("Quelle est la réglementation sur les provisions techniques ?", "rag"),
    ("Comment est calculée la prime nette ?", "rag"),
    ("Qu'est-ce que la PPNA ?", "rag"),
    ("Quels sont les délais de déclaration de sinistre CGA ?", "rag"),
    ("Définition de la franchise en assurance auto", "rag"),
    ("Code des assurances 2002-37 : principales dispositions", "rag"),
    ("Comment est déterminée la marge de solvabilité ?", "rag"),
    ("Qu'est-ce que la tarification actuarielle ?", "rag"),
    ("Qu'est-ce que le CNSS ?", "rag"),
    ("Comment calculer le ratio sinistres sur primes ?", "rag"),
    ("Qu'est-ce que les provisions techniques PSAP/PPNA ?", "rag"),
    ("Quelle est la norme pour le ratio combiné en Tunisie ?", "rag"),
    ("Définition de la branche IRDS", "rag"),
    ("Comment fonctionne l'indemnisation en SANTE ?", "rag"),
    ("Qu'est-ce que le PRC (provision pour risques croissants) ?", "rag"),
    ("Barème de majoration FGA : comment ça marche ?", "rag"),
    ("Règles de tarification branche AUTO selon CGA", "rag"),
    ("Qu'est-ce que la sinistralité nette ?", "rag"),
    ("Délais légaux de règlement sinistre en Tunisie", "rag"),

    # ── kpi ───────────────────────────────────────────────────────────────────
    ("Quel est le taux de résiliation AUTO en 2024 ?", "kpi"),
    ("Donne-moi la prime nette toutes branches T4 2024", "kpi"),
    ("Quelle est la sinistralité branche IRDS ce trimestre ?", "kpi"),
    ("Ratio S/P : quelle est la valeur actuelle en AUTO ?", "kpi"),
    ("Commission moyenne par branche en 2024", "kpi"),
    ("Taux d'impayé SANTE vs objectif", "kpi"),
    ("Performance de la production par rapport à l'an dernier", "kpi"),
    ("Quelle est la prime nette mensuelle en IRDS ?", "kpi"),
    ("Montant total des sinistres payés en 2024", "kpi"),
    ("KPI clés : résiliation, sinistralité, impayes", "kpi"),
    ("Ratio combiné par branche : résultats 2024", "kpi"),
    ("Taux de sinistralité toutes branches confondues", "kpi"),
    ("Quelles sont les métriques de performance AUTO ce mois ?", "kpi"),
    ("Évolution de la prime nette sur 12 mois", "kpi"),
    ("Indicateurs clés : production, risque, rentabilité", "kpi"),
    ("Quelle branche a le meilleur ratio S/P ?", "kpi"),
    ("Commission totale versée aux agents en 2024", "kpi"),
    ("Taux de résiliation IRDS vs cible stratégique", "kpi"),
    ("Prime nette acquise branche SANTE YTD 2024", "kpi"),
    ("Performances financières : toutes métriques KPI", "kpi"),
    ("Montant impayes toutes branches Q3 2024", "kpi"),
    ("Nombre de sinistres déclarés en AUTO ce trimestre", "kpi"),
    ("Ratio sinistres / prime nette branche IRDS", "kpi"),
    ("Évolution du taux d'impayé sur 3 ans", "kpi"),
    ("KPI production : prime, commission, résiliation SANTE", "kpi"),

    # ── sql ───────────────────────────────────────────────────────────────────
    ("Top 5 gouvernorats par montant d'impayes en 2024", "sql"),
    ("Classement des branches par prime nette", "sql"),
    ("Distribution des sinistres par gouvernorat", "sql"),
    ("Liste des 10 meilleurs agents par production", "sql"),
    ("Évolution mensuelle de la prime nette sur 2024", "sql"),
    ("Requête : sinistres par nature et par branche", "sql"),
    ("Top 10 véhicules les plus sinistrés", "sql"),
    ("Combien de polices actives en SANTE au 31/12/2024 ?", "sql"),
    ("Tendance mensuelle du taux d'impayé AUTO", "sql"),
    ("Distribution des primes par tranche de montant", "sql"),
    ("Classement des gouvernorats par sinistralité", "sql"),
    ("Graphique de l'évolution de la résiliation IRDS", "sql"),
    ("Lister les impayes supérieurs à 5000 TND", "sql"),
    ("Top 3 produits par nombre de contrats", "sql"),
    ("Répartition de la production par canal de distribution", "sql"),
    ("Tendance des sinistres sur 3 ans : données mensuelles", "sql"),
    ("Combien de sinistres déclarés par gouvernorat ?", "sql"),
    ("Historique mensuel du ratio S/P par branche", "sql"),
    ("Top 20 clients par prime nette cumulée", "sql"),
    ("Distribution des polices par tranche de bonus-malus", "sql"),
    ("Évolution trimestrielle de la commission toutes branches", "sql"),
    ("Nombre de polices résiliées par mois en 2024", "sql"),
    ("Classement agents : prime nette produite 2024", "sql"),
    ("Données historiques sinistres AUTO : 2019-2024", "sql"),
    ("Top 5 produits IRDS par montant de prime nette", "sql"),
]

# ── Text normalizer ────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase + strip accents + collapse whitespace."""
    text = text.lower()
    # decompose Unicode, remove combining characters (accents)
    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in nfkd if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Build sklearn pipeline ─────────────────────────────────────────────────────

def build_pipeline() -> Any:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import LinearSVC

    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            max_features=20_000,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            max_features=30_000,
        )),
    ])
    # Platt scaling (sigmoid) works well with small datasets and LinearSVC
    clf = CalibratedClassifierCV(
        LinearSVC(C=0.8, max_iter=3000, dual=True),
        cv=5,
        method="sigmoid",
    )
    return Pipeline([("features", features), ("clf", clf)])


# ── Training ───────────────────────────────────────────────────────────────────

def load_eval_store_data() -> list[tuple[str, str]]:
    """Load real labeled data from eval_store (intent + question pairs)."""
    try:
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        import sqlite3, json
        from eval_store import DB_PATH
        if not DB_PATH.exists():
            return []
        con = sqlite3.connect(str(DB_PATH))
        rows = con.execute(
            "SELECT intent, question_len FROM evaluations WHERE intent IS NOT NULL"
        ).fetchall()
        con.close()
        # eval_store only persists question length, not the question text (privacy)
        # So real augmentation requires logging questions separately.
        print(f"[eval_store] {len(rows)} rows found but question text not stored "
              "(only question_len). Use --with-eval once question logging is enabled.")
        return []
    except Exception as e:
        print(f"[eval_store] Could not load: {e}")
        return []


def train(
    extra_data: list[tuple[str, str]] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import make_scorer, f1_score
    import numpy as np

    data = list(_RAW)
    if extra_data:
        data.extend(extra_data)
        print(f"Augmented dataset: {len(_RAW)} synthetic + {len(extra_data)} real = {len(data)} total")
    else:
        print(f"Dataset: {len(data)} examples, {len(set(y for _, y in data))} classes")

    X = [_normalize(q) for q, _ in data]
    y = [label for _, label in data]

    classes = sorted(set(y))
    print(f"Classes ({len(classes)}): {classes}")
    counts = {c: y.count(c) for c in classes}
    print(f"Distribution: {counts}")

    pipeline = build_pipeline()

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline, X, y, cv=cv,
        scoring={
            "accuracy": "accuracy",
            "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        },
        return_train_score=False,
    )

    report: dict[str, Any] = {
        "n_samples":  len(data),
        "n_classes":  len(classes),
        "classes":    classes,
        "cv_accuracy_mean": round(float(np.mean(scores["test_accuracy"])), 4),
        "cv_accuracy_std":  round(float(np.std(scores["test_accuracy"])),  4),
        "cv_f1_macro_mean": round(float(np.mean(scores["test_f1_macro"])), 4),
        "cv_f1_macro_std":  round(float(np.std(scores["test_f1_macro"])),  4),
    }
    print(
        f"\n5-fold CV accuracy : {report['cv_accuracy_mean']:.4f} ± {report['cv_accuracy_std']:.4f}\n"
        f"5-fold CV F1 macro : {report['cv_f1_macro_mean']:.4f} ± {report['cv_f1_macro_std']:.4f}"
    )

    if dry_run:
        print("[dry-run] Model not saved.")
        return report

    # Fit on full dataset
    pipeline.fit(X, y)

    out_path = Path(__file__).resolve().parent / "models" / "intent_classifier.pkl"
    with out_path.open("wb") as fh:
        pickle.dump(
            {
                "pipeline": pipeline,
                "classes":  classes,
                "report":   report,
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"\nSaved -> {out_path}")
    report["saved_to"] = str(out_path)
    return report


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Maghrebia intent classifier")
    parser.add_argument("--with-eval",  action="store_true", help="Merge eval_store data")
    parser.add_argument("--eval-only",  action="store_true", help="Use only eval_store data")
    parser.add_argument("--dry-run",    action="store_true", help="CV only, do not save")
    args = parser.parse_args()

    extra: list[tuple[str, str]] = []
    if args.with_eval or args.eval_only:
        extra = load_eval_store_data()

    base = [] if args.eval_only else None
    train(extra_data=(extra if extra else None), dry_run=args.dry_run)
