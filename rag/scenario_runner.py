from rag.stbtester_rag import STBTesterRAG
from rag.data_loader import load_json_data, ensure_output_dir
import os


def run_scenarios(data_path, output_dir, scenarios, application="Customer_journey"):
    ensure_output_dir(output_dir)
    json_data = load_json_data(data_path)
    rag = STBTesterRAG()
    rag.load_test_cases_data(json_data)
    rag.setup_qa_chain()
    for i, query in enumerate(scenarios, 1):
        print(f"\n🎯 Génération d'un script de test  {i}...")
        result = rag.generate_script(query)
        print(f"\nScénario: {query}")
        print("=" * 50)
        print(result["script"])
        print("=" * 50)
        script_filename = os.path.join(output_dir, f"test_{application}_{i}.py")
        with open(script_filename, "w", encoding="utf-8") as f_script:
            f_script.write(result["script"])
        print(f"\n✅ Script sauvegardé dans : {script_filename}")
        print(f"\n📚 Sources utilisées: {len(result['sources'])}")
        for j, source in enumerate(result["sources"][:2]):
            print(f"  {j+1}. {source['metadata'].get('title', 'N/A')}")
