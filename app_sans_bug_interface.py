import streamlit as st
from rag.stbtester_rag import STBTesterRAG

# from rag.stbtester_rag_openai import STBTesterRAGOpenAI
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

# Affichage de l'historique structuré
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(
            f"<div style='background-color:#e8e8e8; padding:8px; border-radius:6px; margin-bottom:2px; font-size:14px;'>Utilisateur : {msg['content']}</div>",
            unsafe_allow_html=True,
        )
    else:
        import re

        content = msg["content"]
        # Sépare les blocs de code et le texte explicatif
        code_block_pattern = r"```python(.*?)```"
        last_end = 0
        for match in re.finditer(
            code_block_pattern, content, re.DOTALL | re.IGNORECASE
        ):
            # Affiche le texte avant le bloc code (explication)
            explanation = content[last_end : match.start()].strip()
            if explanation:
                # Nettoie et filtre comme pour le raisonnement
                cleaned_explanation = explanation
                cleaned_explanation = re.sub(
                    r"^(#+) ?(.*)$", r"\2", cleaned_explanation, flags=re.MULTILINE
                )
                cleaned_explanation = re.sub(
                    r"<h[1-6][^>]*>.*?</h[1-6]>",
                    "",
                    cleaned_explanation,
                    flags=re.DOTALL,
                )
                cleaned_explanation = re.sub(
                    r"<a[^>]*>.*?</a>", "", cleaned_explanation, flags=re.DOTALL
                )
                cleaned_explanation = re.sub(
                    r"<span[^>]*>.*?</span>", "", cleaned_explanation, flags=re.DOTALL
                )
                cleaned_explanation = re.sub(
                    r"<svg[^>]*>.*?</svg>", "", cleaned_explanation, flags=re.DOTALL
                )
                cleaned_explanation = re.sub(r"<[^>]+>", "", cleaned_explanation)
                st.markdown(
                    f"<div style='background-color:#f0f2f6; padding:8px; border-radius:6px; margin-bottom:2px; font-size:14px; font-family: 'Segoe UI', Arial, sans-serif;'><pre style='font-size:12px; font-family:monospace; background:transparent; border:none; margin:0; padding:0; white-space:pre-wrap;'>{cleaned_explanation}</pre></div>",
                    unsafe_allow_html=True,
                )
            # Affiche le bloc code
            code = match.group(1).strip()
            if code:
                st.code(code, language="python")
            last_end = match.end()
        # Affiche le texte après le dernier bloc code
        rest = content[last_end:].strip()
        if rest:
            cleaned_rest = rest
            cleaned_rest = re.sub(
                r"^(#+) ?(.*)$", r"\2", cleaned_rest, flags=re.MULTILINE
            )
            cleaned_rest = re.sub(
                r"<h[1-6][^>]*>.*?</h[1-6]>", "", cleaned_rest, flags=re.DOTALL
            )
            cleaned_rest = re.sub(r"<a[^>]*>.*?</a>", "", cleaned_rest, flags=re.DOTALL)
            cleaned_rest = re.sub(
                r"<span[^>]*>.*?</span>", "", cleaned_rest, flags=re.DOTALL
            )
            cleaned_rest = re.sub(
                r"<svg[^>]*>.*?</svg>", "", cleaned_rest, flags=re.DOTALL
            )
            cleaned_rest = re.sub(r"<[^>]+>", "", cleaned_rest)
            st.markdown(
                f"<div style='background-color:#f0f2f6; padding:8px; border-radius:6px; margin-bottom:2px; font-size:14px; font-family: 'Segoe UI', Arial, sans-serif;'><pre style='font-size:12px; font-family:monospace; background:transparent; border:none; margin:0; padding:0; white-space:pre-wrap;'>{cleaned_rest}</pre></div>",
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
    # Affiche immédiatement la question de l'utilisateur dans l'historique
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(
        f"<div style='background-color:#e8e8e8; padding:8px; border-radius:6px; margin-bottom:2px;'>Utilisateur : {prompt}</div>",
        unsafe_allow_html=True,
    )
    # --- Génération de script STBTester via RAG pour toute question ---
    try:
        code_placeholder = st.empty()
        script = ""
        reasoning = ""
        # Ajout de l'historique conversationnel dans le prompt
        # Prend les 3 derniers échanges user/assistant (jusqu'à 6 messages)
        last_msgs = [
            m
            for m in st.session_state.messages[:-1]
            if m["role"] in ("user", "assistant")
        ]
        # On veut les 3 dernières paires (user/assistant), donc on part de la fin
        pairs = []
        i = len(last_msgs) - 1
        while i >= 0 and len(pairs) < 3:
            if (
                last_msgs[i]["role"] == "assistant"
                and i > 0
                and last_msgs[i - 1]["role"] == "user"
            ):
                pairs.append((last_msgs[i - 1], last_msgs[i]))
                i -= 2
            else:
                i -= 1
        # On remet dans l'ordre chronologique
        pairs = pairs[::-1]
        history = []
        for user_msg, assistant_msg in pairs:
            history.append(f"Utilisateur : {user_msg['content']}")
            history.append(f"Assistant : {assistant_msg['content']}")
        history_text = "\n".join(history)
        full_prompt = (
            f"{history_text}\nUtilisateur : {prompt}" if history_text else prompt
        )
        with st.spinner("Le modèle réfléchit et génère le script..."):
            # --- Streaming mot à mot uniquement sur la réponse réelle (hors <think>) ---
            display_placeholder = st.empty()
            streamed_real_response = ""
            in_think = False
            for chunk in rag.stream_generate_script(full_prompt):
                script += chunk
                # Ajoute chunk à la partie "réponse réelle" seulement si on n'est pas dans <think>
                i = 0
                while i < len(chunk):
                    # Détecte début <think>
                    if not in_think and chunk[i : i + 7].lower() == "<think>":
                        in_think = True
                        i += 7
                        continue
                    # Détecte fin </think>
                    if in_think and chunk[i : i + 8].lower() == "</think>":
                        in_think = False
                        i += 8
                        continue
                    if not in_think:
                        streamed_real_response += chunk[i]
                        # Affiche en temps réel mot à mot
                        display_placeholder.markdown(
                            f"<div style='background-color:#f0f2f6; padding:8px; border-radius:6px; margin-bottom:2px; font-size:14px; font-family: 'Segoe UI', Arial, sans-serif;'><pre style='font-size:12px; font-family:monospace; background:transparent; border:none; margin:0; padding:0; white-space:pre-wrap;'>{streamed_real_response}</pre></div>",
                            unsafe_allow_html=True,
                        )
                    i += 1
        # --- Affichage raisonnement/thinking (think) dans l'expander uniquement si <think> et </think> sont présents ---
        if "<think>" in script and "</think>" in script:
            start_idx = script.lower().find("<think>") + len("<think>")
            end_idx = script.lower().find("</think>")
            reasoning = script[start_idx:end_idx].strip()
            # Nettoyage raisonnement
            import re

            cleaned_reasoning = reasoning
            cleaned_reasoning = re.sub(
                r"^(#+) ?(.*)$", r"\2", cleaned_reasoning, flags=re.MULTILINE
            )
            cleaned_reasoning = re.sub(
                r"<h[1-6][^>]*>.*?</h[1-6]>", "", cleaned_reasoning, flags=re.DOTALL
            )
            cleaned_reasoning = re.sub(
                r"<a[^>]*>.*?</a>", "", cleaned_reasoning, flags=re.DOTALL
            )
            cleaned_reasoning = re.sub(
                r"<span[^>]*>.*?</span>", "", cleaned_reasoning, flags=re.DOTALL
            )
            cleaned_reasoning = re.sub(
                r"<svg[^>]*>.*?</svg>", "", cleaned_reasoning, flags=re.DOTALL
            )
            cleaned_reasoning = re.sub(r"<[^>]+>", "", cleaned_reasoning)
            if len(cleaned_reasoning) > 1000:
                cleaned_reasoning = cleaned_reasoning[:1000] + "... [contenu tronqué]"
            # Ajout à l'historique : raisonnement et réponse (hors <think>)
            import re

            script_no_think = re.sub(
                r"<think>.*?</think>", "", script, flags=re.DOTALL | re.IGNORECASE
            )
            script_no_think = script_no_think.strip()
            st.session_state.messages.append(
                {"role": "assistant", "content": script_no_think}
            )
            with open(history_path, "w", encoding="utf-8") as f_conv:
                json.dump(
                    st.session_state.messages, f_conv, ensure_ascii=False, indent=2
                )
            with st.expander("Afficher/masquer le raisonnement du modèle (optionnel)"):
                st.info(cleaned_reasoning)
        else:
            # Si pas de balise <think>...</think>, ajoute la réponse complète dans l'historique
            st.session_state.messages.append(
                {"role": "assistant", "content": script.strip()}
            )
            with open(history_path, "w", encoding="utf-8") as f_conv:
                json.dump(
                    st.session_state.messages, f_conv, ensure_ascii=False, indent=2
                )
    except Exception as e:
        response = f"Erreur lors de la génération du script : {e}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        with open(history_path, "w", encoding="utf-8") as f_conv:
            json.dump(st.session_state.messages, f_conv, ensure_ascii=False, indent=2)

# st.info(
#     "Ce chatbot génère des scripts de test STBTester à partir de scénarios en langage naturel grâce à votre pipeline RAG."
# )
