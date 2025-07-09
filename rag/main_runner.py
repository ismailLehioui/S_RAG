import os
from rag.scenario_runner import run_scenarios
from rag.data_loader import load_scenarios_from_file


def run_rag():
    application = "youtube"
    data_path = "data/new_data.json"
    output_dir = f"generated_tests/{application}"
    scenario_file = "scenarios/yaml/scenario.yaml"  # Peut être .yaml, .json, .txt
    # scenarios = [
    #     """Aller sur Live TV\nAccéder au menu principal\nNaviguer vers STREAMING et sélectionner Netflix\nConfirmer la sélection de Netflix\nAttendre le chargement de Netflix (10 secondes)\nLancer la première vidéo recommandée\nRépéter pendant 1 heure : Lecture 10 min, Pause, Lecture\nQuitter Netflix proprement
    #     """
    # ]

    scenarios = load_scenarios_from_file(scenario_file)
    run_scenarios(data_path, output_dir, scenarios, application)

    # Pour tester SANS RAG (LLM pur), passer no_rag=True
    # scenarios = load_scenarios_from_file(scenario_file)
    # run_scenarios(
    #     data_path,
    #     output_dir,
    #     scenarios,
    #     application,
    #     with_context=False,
    #     no_rag=True,
    # )
