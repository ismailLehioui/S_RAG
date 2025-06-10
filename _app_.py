# avec thinking

# import streamlit as st
# from rag.stbtester_rag import STBTesterRAG
# import json
# import os

# st.set_page_config(page_title="Chatbot STBTester", page_icon="🤖")
# st.title("Chatbot RAG pour l'automatisation de tests STBTester")

# # Initialisation du moteur RAG (chargement des cas de test une seule fois)
# if "rag" not in st.session_state:
#     rag = STBTesterRAG()
#     # Charge les cas de test depuis le JSON principal
#     with open("data/all_data.json", "r", encoding="utf-8") as f:
#         json_data = json.load(f)
#     rag.ensure_vectorstore(json_data)  # Embedding seulement si la base n'existe pas
#     rag.setup_qa_chain()
#     st.session_state.rag = rag
# else:
#     rag = st.session_state.rag

# # Historique de la conversation
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Affichage de l'historique
# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# # Entrée utilisateur
# prompt = st.chat_input("Décrivez un scénario de test ou posez une question...")

# if prompt:
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     try:
#         # Génération du script via le moteur RAG en streaming
#         code_placeholder = st.empty()
#         script = ""
#         for chunk in rag.stream_generate_script(prompt):
#             script += chunk
#             # Extraction du code entre ```python ... ```
#             code_only = script
#             if "```python" in script:
#                 start = script.find("```python") + len("```python")
#                 end = script.find("```", start)
#                 if end != -1:
#                     code_only = script[start:end].strip()
#                 else:
#                     code_only = script[start:].strip()
#             # Filtrage renforcé : supprime toute ligne contenant 'think', 'réflexion', 'reflection' (même dans les commentaires)
#             filtered_code = "\n".join(
#                 line
#                 for line in code_only.splitlines()
#                 if all(
#                     token not in line.lower()
#                     for token in ["think", "réflexion", "reflection"]
#                 )
#             )
#             code_placeholder.code(filtered_code, language="python")
#         response = filtered_code
#         # Sauvegarde automatique du script généré dans un fichier
#         output_dir = "generated_tests/streamlit"
#         os.makedirs(output_dir, exist_ok=True)
#         script_filename = os.path.join(
#             output_dir, f"test_streamlit_{len(os.listdir(output_dir))+1}.py"
#         )
#         with open(script_filename, "w", encoding="utf-8") as f_script:
#             f_script.write(filtered_code)
#         st.success(f"Script sauvegardé automatiquement dans : {script_filename}")
#     except Exception as e:
#         response = f"❌ Erreur lors de la génération du script : {e}"
#     st.session_state.messages.append({"role": "assistant", "content": response})
#     st.chat_message("assistant").write(response)

# st.info(
#     "Ce chatbot génère des scripts de test STBTester à partir de scénarios en langage naturel grâce à votre pipeline RAG."
# )
