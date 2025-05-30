import os
from rag.scenario_runner import run_scenarios
from rag.data_loader import load_scenarios_from_file


def run_rag():
    application = "youtube"
    data_path = "data/all_data.json"
    output_dir = f"generated_tests/{application}"
    scenario_file = (
        "scenarios/yaml/scenarios_Amazon.yaml"  # Peut être .yaml, .json, .txt
    )
    scenarios = [
        "Aller sur Live TV\nAller dans STREAMING et sélectionner Amazon\nAttendre le chargement d'Amazon\nLancer une vidéo Amazon\nAttendre 10 minutes\nQuitter Amazon proprement\nRetourner sur Live TV"
    ]
    # scenarios = load_scenarios_from_file(scenario_file)
    run_scenarios(data_path, output_dir, scenarios, application)
