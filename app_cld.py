import streamlit as st
from rag.stbtester_rag import STBTesterRAG

# from rag.stbtester_rag_openai import STBTesterRAGOpenAI
import json
import os
import uuid
import re

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

# --- Gestion multi-conversations dans le sidebar ---
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
    selected_idx = (
        st.radio(
            "Conversations :",
            options=list(range(len(conv_titles))),
            format_func=lambda i: conv_titles[i] if conv_titles else "",
            index=st.session_state.get("selected_conv_idx", 0) if conv_titles else 0,
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
    with open("data/new_data.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    rag.ensure_vectorstore(json_data)
    rag.setup_qa_chain()
    st.session_state.rag = rag
else:
    rag = st.session_state.rag


# Fonctions utilitaires pour parser le contenu
def extract_thinking_and_content(text):
    """
    Extrait le contenu <think> et le contenu principal
    Returns: (thinking_content, main_content)
    """
    thinking_pattern = r"<think>(.*?)</think>"
    thinking_matches = re.findall(thinking_pattern, text, re.DOTALL)

    # Supprimer tout le contenu <think>...</think> pour obtenir le contenu principal
    main_content = re.sub(thinking_pattern, "", text, flags=re.DOTALL).strip()

    thinking_content = "\n".join(thinking_matches) if thinking_matches else ""

    return thinking_content, main_content


def extract_python_code(text):
    """
    Extrait uniquement le code Python du texte
    """
    if "```python" in text:
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        if end != -1:
            code = text[start:end].strip()
        else:
            code = text[start:].strip()

        # Filtrer les lignes de réflexion
        filtered_code = "\n".join(
            line
            for line in code.splitlines()
            if not any(
                token in line.lower() for token in ["think", "réflexion", "reflection"]
            )
        )
        return filtered_code
    return ""


def is_code_generation_request(prompt):
    """
    Détermine si la demande concerne la génération de code
    """
    code_keywords = [
        "script",
        "code",
        "test",
        "génère",
        "créer",
        "écrire",
        "fonction",
        "automatiser",
        "scenario",
        "stbtester",
        "python",
    ]
    return any(keyword in prompt.lower() for keyword in code_keywords)


# Affichage de l'historique structuré
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(
            f"<div style='background-color:#e8e8e8; padding:8px; border-radius:6px; margin-bottom:2px; font-size:14px;'>Utilisateur : {msg['content']}</div>",
            unsafe_allow_html=True,
        )
    else:
        # Nettoyage du contenu pour l'affichage
        cleaned = msg["content"]
        # Supprime les titres markdown et balises HTML
        cleaned = re.sub(r"^(#+) ?(.*)$", r"\2", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"<h[1-6][^>]*>.*?</h[1-6]>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<a[^>]*>.*?</a>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<span[^>]*>.*?</span>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<svg[^>]*>.*?</svg>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<[^>]+>", "", cleaned)

        # Vérifier s'il y a du code Python à afficher
        python_code = extract_python_code(msg["content"])
        if python_code:
            st.code(python_code, language="python")
        else:
            st.markdown(
                f"<div style='background-color:#f0f2f6; padding:8px; border-radius:6px; margin-bottom:2px; font-size:14px; font-family: \"Segoe UI\", Arial, sans-serif;'><pre style='font-size:12px; font-family:monospace; background:transparent; border:none; margin:0; padding:0; white-space:pre-wrap;'>{cleaned}</pre></div>",
                unsafe_allow_html=True,
            )

    # Séparation visuelle entre chaque échange
    if i < len(st.session_state.messages) - 1:
        st.markdown(
            "<hr style='margin:8px 0; border: none; border-top: 1px solid #ccc; background: none; height: 1px;'>",
            unsafe_allow_html=True,
        )

# Entrée utilisateur
prompt = st.chat_input("Décrivez un scénario de test ou posez une question...")

if prompt:
    # Affiche immédiatement la question de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(
        f"<div style='background-color:#e8e8e8; padding:8px; border-radius:6px; margin-bottom:2px;'>Utilisateur : {prompt}</div>",
        unsafe_allow_html=True,
    )

    # Déterminer le type de demande
    is_code_request = is_code_generation_request(prompt)

    try:
        # Variables pour stocker le contenu
        full_response = ""
        thinking_content = ""
        display_content = ""
        current_thinking = ""
        in_thinking = False

        # Créer les placeholders pour l'affichage
        if is_code_request:
            code_placeholder = st.empty()
            status_placeholder = st.empty()
        else:
            response_placeholder = st.empty()
            status_placeholder = st.empty()

        status_placeholder.info("🤔 Le modèle réfléchit...")

        # Streaming de la réponse
        for chunk in rag.stream_generate_script(prompt):
            full_response += chunk

            # Détecter le début et la fin des balises <think>
            if "<think>" in chunk:
                in_thinking = True
                # Extraire la partie avant <think> si elle existe
                before_think = chunk.split("<think>")[0]
                if before_think.strip():
                    display_content += before_think
                continue

            if "</think>" in chunk:
                in_thinking = False
                # Extraire la partie après </think> si elle existe
                after_think = chunk.split("</think>")[-1]
                if after_think.strip():
                    display_content += after_think
                continue

            # Si on est dans une section <think>, l'ajouter au thinking
            if in_thinking:
                current_thinking += chunk
                continue

            # Sinon, l'ajouter au contenu d'affichage
            display_content += chunk

            # Mise à jour de l'affichage en temps réel
            if is_code_request:
                # Pour les demandes de code, extraire et afficher le code Python
                python_code = extract_python_code(display_content)
                if python_code.strip():
                    code_placeholder.code(python_code, language="python")
                    status_placeholder.success("✅ Code généré")
                else:
                    status_placeholder.info("🔄 Génération en cours...")
            else:
                # Pour les questions normales, afficher le texte directement
                if display_content.strip():
                    # Nettoyer le contenu pour l'affichage
                    cleaned_display = display_content.strip()
                    response_placeholder.markdown(
                        f"<div style='background-color:#f0f2f6; padding:8px; border-radius:6px; font-size:14px; font-family: \"Segoe UI\", Arial, sans-serif;'>{cleaned_display}</div>",
                        unsafe_allow_html=True,
                    )
                    status_placeholder.success("✅ Réponse générée")

        # Traitement final après le streaming
        thinking_final, main_content_final = extract_thinking_and_content(full_response)
        if current_thinking:
            thinking_final = current_thinking + "\n" + thinking_final

        # Déterminer la réponse finale à sauvegarder
        if is_code_request:
            python_code = extract_python_code(main_content_final)
            if python_code.strip():
                final_response = python_code

                # Sauvegarder le fichier
                output_dir = "generated_tests/streamlit"
                os.makedirs(output_dir, exist_ok=True)
                script_filename = os.path.join(
                    output_dir, f"test_streamlit_{len(os.listdir(output_dir))+1}.py"
                )
                with open(script_filename, "w", encoding="utf-8") as f_script:
                    f_script.write(python_code)
                st.success(f"✅ Script sauvegardé dans : {script_filename}")
            else:
                final_response = (
                    "Aucun code Python valide généré. Reformulez votre demande."
                )
                st.error(final_response)
        else:
            final_response = (
                main_content_final.strip()
                if main_content_final.strip()
                else display_content.strip()
            )

        # Nettoyage du contenu pour l'affichage (suppression titres markdown et balises HTML)
        cleaned = final_response
        cleaned = re.sub(r"^(#+) ?(.*)$", r"\2", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"<h[1-6][^>]*>.*?</h[1-6]>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<a[^>]*>.*?</a>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<span[^>]*>.*?</span>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<svg[^>]*>.*?</svg>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<[^>]+>", "", cleaned)
        st.markdown(
            f"<div style='background-color:#f0f2f6; padding:8px; border-radius:6px; margin-bottom:2px; font-size:14px; font-family: 'Segoe UI', Arial, sans-serif;'><pre style='font-size:12px; font-family:monospace; background:transparent; border:none; margin:0; padding:0; white-space:pre-wrap;'>{cleaned}</pre></div>",
            unsafe_allow_html=True,
        )

        # Afficher le raisonnement s'il existe
        if thinking_final.strip():
            with st.expander(
                "🧠 Afficher/masquer le raisonnement du modèle (optionnel)"
            ):
                st.info(thinking_final.strip())

        # Effacer le status
        status_placeholder.empty()

    except Exception as e:
        final_response = f"❌ Erreur lors de la génération : {e}"
        st.error(final_response)

    # Sauvegarder dans l'historique
    st.session_state.messages.append({"role": "assistant", "content": final_response})

    # Sauvegarder l'historique de conversation
    with open(history_path, "w", encoding="utf-8") as f_conv:
        json.dump(st.session_state.messages, f_conv, ensure_ascii=False, indent=2)
