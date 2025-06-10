import streamlit as st
from rag.stbtester_rag import STBTesterRAG
import json
import os
import uuid

st.set_page_config(page_title="Chatbot STBTester", page_icon="🤖")
st.title("Génération de tests STBTester")

# --- Conversation Management ---
CONV_DIR = "generated_tests/streamlit"
CONV_INDEX = os.path.join(CONV_DIR, "conversations_index.json")

# Helper to load/save conversation index


def load_conversations_index():
    if os.path.exists(CONV_INDEX):
        with open(CONV_INDEX, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return []


def save_conversations_index(index):
    with open(CONV_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


# Initialize conversations index in session state
if "conversations_index" not in st.session_state:
    st.session_state.conversations_index = load_conversations_index()

# --- SIDEBAR UI ---
st.sidebar.title("💬 Conversations")

# List of conversations (show name or id)
conv_options = [
    f"{conv.get('name', 'Conversation')} ({conv['id'][:8]})"
    for conv in st.session_state.conversations_index
]
conv_ids = [conv["id"] for conv in st.session_state.conversations_index]

# Select current conversation
if "current_conv_id" not in st.session_state:
    if conv_ids:
        st.session_state.current_conv_id = conv_ids[0]
    else:
        st.session_state.current_conv_id = None

selected = st.sidebar.selectbox(
    "Sélectionnez une conversation :",
    options=conv_options,
    index=(
        conv_ids.index(st.session_state.current_conv_id)
        if st.session_state.current_conv_id in conv_ids
        else 0 if conv_ids else None
    ),
    key="sidebar_selectbox",
)

# Map selection back to conv_id
if conv_options:
    sel_idx = conv_options.index(selected)
    selected_conv_id = conv_ids[sel_idx]
else:
    selected_conv_id = None

# Switch conversation if changed
if selected_conv_id and selected_conv_id != st.session_state.get("current_conv_id"):
    st.session_state.current_conv_id = selected_conv_id
    # Load messages for this conversation
    conv_path = os.path.join(CONV_DIR, f"conversation_{selected_conv_id}.json")
    if os.path.exists(conv_path):
        with open(conv_path, "r", encoding="utf-8") as f:
            st.session_state.messages = json.load(f)
    else:
        st.session_state.messages = []
    st.rerun()


# Button to create new conversation
def create_new_conversation():
    new_id = str(uuid.uuid4())
    new_name = st.sidebar.text_input(
        "Nom de la nouvelle conversation",
        value="Nouvelle conversation",
        key="new_conv_name",
    )
    if st.sidebar.button("Créer", key="create_conv_btn"):
        # Add to index
        st.session_state.conversations_index.append({"id": new_id, "name": new_name})
        save_conversations_index(st.session_state.conversations_index)
        # Create empty history file
        conv_path = os.path.join(CONV_DIR, f"conversation_{new_id}.json")
        with open(conv_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        st.session_state.current_conv_id = new_id
        st.session_state.messages = []
        st.rerun()


with st.sidebar.expander("Créer une nouvelle conversation"):
    create_new_conversation()

# Button to delete current conversation
if st.session_state.current_conv_id:
    if st.sidebar.button("🗑️ Supprimer cette conversation", key="delete_conv_btn"):
        # Remove from index
        st.session_state.conversations_index = [
            c
            for c in st.session_state.conversations_index
            if c["id"] != st.session_state.current_conv_id
        ]
        save_conversations_index(st.session_state.conversations_index)
        # Delete file
        conv_path = os.path.join(
            CONV_DIR, f"conversation_{st.session_state.current_conv_id}.json"
        )
        if os.path.exists(conv_path):
            os.remove(conv_path)
        # Switch to another or clear
        if st.session_state.conversations_index:
            st.session_state.current_conv_id = st.session_state.conversations_index[0][
                "id"
            ]
            conv_path = os.path.join(
                CONV_DIR, f"conversation_{st.session_state.current_conv_id}.json"
            )
            if os.path.exists(conv_path):
                with open(conv_path, "r", encoding="utf-8") as f:
                    st.session_state.messages = json.load(f)
            else:
                st.session_state.messages = []
        else:
            st.session_state.current_conv_id = None
            st.session_state.messages = []
        st.rerun()

# --- END SIDEBAR ---

# --- Gestion multi-conversations dans le sidebar ---
import uuid

CONV_DIR = "generated_tests/streamlit"
CONV_INDEX_PATH = os.path.join(CONV_DIR, "conversations_index.json")

# Charger ou initialiser l'index des conversations
if not os.path.exists(CONV_DIR):
    os.makedirs(CONV_DIR, exist_ok=True)
if os.path.exists(CONV_INDEX_PATH):
    with open(CONV_INDEX_PATH, "r", encoding="utf-8") as f:
        conversations_index = json.load(f)
else:
    conversations_index = []
    with open(CONV_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(conversations_index, f)

# Sidebar : liste des conversations et création
with st.sidebar:
    st.header("💬 Conversations")
    conv_titles = [
        conv.get("title", conv.get("name", "Conversation"))
        for conv in conversations_index
    ]
    conv_ids = [conv["id"] for conv in conversations_index]
    if len(conv_titles) == 0:
        st.write("Aucune conversation. Créez-en une !")
    safe_idx = st.session_state.get("selected_conv_idx", 0)
    if not conv_titles or safe_idx < 0 or safe_idx >= len(conv_titles):
        safe_idx = 0
    selected_idx = (
        st.radio(
            "Conversations :",
            options=list(range(len(conv_titles))),
            format_func=lambda i: conv_titles[i] if conv_titles else "",
            index=safe_idx,
            key="conv_radio",
        )
        if conv_titles
        else None
    )
    if st.button("➕ Nouvelle conversation"):
        new_id = str(uuid.uuid4())
        new_title = f"Conversation {len(conversations_index)+1}"
        conversations_index.append({"id": new_id, "title": new_title})
        with open(CONV_INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(conversations_index, f, ensure_ascii=False, indent=2)
        # Crée un fichier vide pour la nouvelle conversation
        with open(
            os.path.join(CONV_DIR, f"conversation_{new_id}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump([], f)
        st.session_state.selected_conv_idx = len(conversations_index) - 1
        st.rerun()
    # Switch de conversation
    if conv_titles and selected_idx is not None:
        st.session_state.selected_conv_idx = selected_idx
        selected_conv_id = conv_ids[selected_idx]
        history_path = os.path.join(CONV_DIR, f"conversation_{selected_conv_id}.json")
    else:
        selected_conv_id = None
        history_path = os.path.join(CONV_DIR, "conversation_history.json")

# Chargement de l'historique de la conversation sélectionnée
if (
    "messages" not in st.session_state
    or st.session_state.get("last_loaded_conv") != selected_conv_id
):
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f_conv:
            st.session_state.messages = json.load(f_conv)
    else:
        st.session_state.messages = []
    st.session_state.last_loaded_conv = selected_conv_id

# Initialisation du moteur RAG (chargement des cas de test une seule fois)
if "rag" not in st.session_state:
    rag = STBTesterRAG()
    with open("data/all_data.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    rag.ensure_vectorstore(json_data)
    rag.setup_qa_chain()
    st.session_state.rag = rag
else:
    rag = st.session_state.rag

# Affichage de l'historique structuré
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(f"**Utilisateur :** {msg['content']}")
    else:
        # Affichage assistant : ne met pas en gras les lignes commençant par 'Étape'
        lines = msg["content"].splitlines()
        formatted = []
        for line in lines:
            if line.strip().startswith("Étape"):
                formatted.append(line)  # Pas de gras
            else:
                formatted.append(line)
        st.markdown(
            f"<div style='background-color:#f0f2f6; padding:8px; border-radius:6px;'><b>Assistant :</b><br>{'<br>'.join(formatted)}</div>",
            unsafe_allow_html=True,
        )
    # Séparation visuelle entre chaque échange
    if i < len(st.session_state.messages) - 1:
        st.markdown("<hr style='margin:8px 0;'>", unsafe_allow_html=True)

# Entrée utilisateur
prompt = st.chat_input("Décrivez un scénario de test ou posez une question...")

if prompt:
    # Cas spécial : question sur la dernière question posée
    if "dernier question" in prompt.lower():
        previous_questions = [
            m["content"] for m in st.session_state.messages if m["role"] == "user"
        ]
        if previous_questions:
            last_question = previous_questions[-1]
            response = f"Votre dernière question était :\n\n{last_question}"
        else:
            response = "Je n'ai pas trouvé de question précédente dans l'historique."
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"**Utilisateur :** {prompt}")
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(response)
        with open(history_path, "w", encoding="utf-8") as f_conv:
            json.dump(st.session_state.messages, f_conv, ensure_ascii=False, indent=2)
        st.stop()
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"**Utilisateur :** {prompt}")
    keywords = [
        "script",
        "test",
        "stbtester",
        "python",
        "automatisation",
        "génère",
        "écris",
        "générer",
        "automatiser",
    ]
    if any(kw in prompt.lower() for kw in keywords):
        # --- Génération de script STBTester ---
        try:
            code_placeholder = st.empty()
            script = ""
            reasoning = ""
            with st.spinner("Le modèle réfléchit ..."):
                for chunk in rag.stream_generate_script(prompt):
                    script += chunk
                    code_only = ""
                    if "```python" in script:
                        start = script.find("```python") + len("```python")
                        end = script.find("```", start)
                        if end != -1:
                            code_only = script[start:end].strip()
                            reasoning = (
                                script[: start - len("```python")].strip()
                                + "\n"
                                + script[end + len("```") :].strip()
                            ).strip()
                        else:
                            code_only = script[start:].strip()
                            reasoning = script[: start - len("```python")].strip()
                    else:
                        reasoning = script
                    filtered_code = "\n".join(
                        line
                        for line in code_only.splitlines()
                        if all(
                            token not in line.lower()
                            for token in ["think", "réflexion", "reflection"]
                        )
                    )
                    code_placeholder.code(filtered_code, language="python")
            response = (
                filtered_code
                if filtered_code.strip()
                else "Aucun code Python valide n'a été généré. Reformulez votre demande."
            )
            if filtered_code.strip():
                output_dir = "generated_tests/streamlit"
                os.makedirs(output_dir, exist_ok=True)
                script_filename = os.path.join(
                    output_dir, f"test_streamlit_{len(os.listdir(output_dir))+1}.py"
                )
                with open(script_filename, "w", encoding="utf-8") as f_script:
                    f_script.write(filtered_code)
                st.success(
                    f"Script sauvegardé automatiquement dans : {script_filename}"
                )
            # Affichage du raisonnement/thinking dans un expander si présent (mais pas dans l'historique)
            if reasoning.strip():
                with st.expander(
                    "Afficher/masquer le raisonnement du modèle (optionnel)"
                ):
                    st.info(reasoning.strip())
        except Exception as e:
            response = f"Erreur lors de la génération du script : {e}"
        # Ne sauvegarde que le code/texte visible dans l'historique
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
    else:
        # --- Mode conversationnel classique avec streaming ---
        try:
            response_placeholder = st.empty()
            stream_response = ""
            thinking = ""
            with st.spinner("Le modèle réfléchit..."):
                if hasattr(rag.llm, "stream"):
                    for chunk in rag.llm.stream(prompt):
                        stream_response += chunk
                        # Séparation du thinking si balises <think> détectées
                        if "<think>" in stream_response:
                            start = stream_response.find("<think>") + len("<think>")
                            end = stream_response.find("</think>", start)
                            if end != -1:
                                thinking = stream_response[start:end].strip()
                                visible_response = (
                                    stream_response[: start - len("<think>")].strip()
                                    + stream_response[end + len("</think>") :].strip()
                                )
                            else:
                                visible_response = stream_response.replace(
                                    "<think>", ""
                                )
                        else:
                            visible_response = stream_response
                        response_placeholder.markdown(visible_response)
                else:
                    stream_response = rag.llm(prompt)
                    response_placeholder.markdown(stream_response)
            response = (
                visible_response if "visible_response" in locals() else stream_response
            )
            # Affichage du thinking dans un expander si présent (mais pas dans l'historique)
            if thinking.strip():
                with st.expander(
                    "Afficher/masquer le raisonnement du modèle (optionnel)"
                ):
                    st.info(thinking.strip())
        except Exception as e:
            response = f"Erreur lors de la génération de la réponse : {e}"
        # Ne sauvegarde que la réponse visible dans l'historique
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
    # Sauvegarde de l'historique de conversation après chaque échange (sans reasoning/thinking)
    with open(history_path, "w", encoding="utf-8") as f_conv:
        json.dump(st.session_state.messages, f_conv, ensure_ascii=False, indent=2)

# st.info(
#     "Ce chatbot génère des scripts de test STBTester à partir de scénarios en langage naturel grâce à votre pipeline RAG."
# )
