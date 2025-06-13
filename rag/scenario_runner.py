from rag.stbtester_rag import STBTesterRAG

# from rag.stbtester_rag_openai import STBTesterRAGOpenAI

from rag.data_loader import load_json_data, ensure_output_dir
import os


def run_scenarios(
    data_path,
    output_dir,
    scenarios,
    application="Customer_journey",
    with_context=True,
    # no_rag=False,
):
    ensure_output_dir(output_dir)
    json_data = load_json_data(data_path)
    rag = STBTesterRAG()
    # rag.load_test_cases_data(json_data)
    rag.ensure_vectorstore(json_data, data_path=data_path)
    rag.setup_qa_chain()
    for i, query in enumerate(scenarios, 1):
        print(f"\nScénario: {query}")
        print("=" * 50)
        print("Script généré (streaming) :\n")
        script = ""
        for chunk in rag.stream_generate_script(query):
            # Filtrer les tokens ou blocs contenant '<think>' (insensible à la casse)
            if "<think>" in chunk.lower():
                continue
            print(chunk, end="", flush=True)
            script += chunk
        print("\n" + "=" * 50)
        # Extract only the code between ```python and the next ```
        code_only = script
        if "```python" in script:
            start = script.find("```python") + len("```python")
            end = script.find("```", start)
            if end != -1:
                code_only = script[start:end].strip()
            else:
                code_only = script[start:].strip()
        # Remove any lines containing '<think>' or similar thinking tokens before saving
        filtered_script = "\n".join(
            line
            for line in code_only.splitlines()
            if all(
                token not in line.lower()
                for token in ["<think>", "<réflexion>", "<reflection>"]
            )
        )
        script_filename = os.path.join(output_dir, f"test_{application}_{i}.py")
        with open(script_filename, "w", encoding="utf-8") as f_script:
            f_script.write(filtered_script)
        print(f"\n Script sauvegardé dans : {script_filename}")
        # Les sources ne sont pas disponibles en streaming, donc on ne les affiche pas ici
