import re
import logging
import os
import hashlib
from typing import List, Dict, Any
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from chromadb.config import Settings
from langchain.memory import ConversationSummaryMemory, ConversationBufferWindowMemory

# Pour VectorStoreRetrieverMemory, on importe FAISS et HuggingFaceEmbeddings si besoin
try:
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.memory import VectorStoreRetrieverMemory

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class STBTesterRAG:
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model_name: str = "qwen3:8b",
        persist_directory: str = "./chroma_db",
        memory_type: str = "none",  # 'none', 'summary', 'window', 'vectorstore'
        memory_kwargs: dict = None,  # paramètres pour la mémoire
    ):
        self.ollama_base_url = ollama_base_url
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.memory_type = memory_type
        self.memory_kwargs = memory_kwargs or {}
        self.memory = None
        self._init_embeddings()
        self._init_llm()
        self._init_vectorstore()
        self._init_memory()  # Ajouté ici
        self._init_qa_chain()

    def _init_embeddings(self):
        try:
            self.embeddings = OllamaEmbeddings(
                base_url=self.ollama_base_url,
                model="nomic-embed-text",
            )
            logger.info("✅ Embeddings initialisés")
        except Exception as e:
            logger.error(f" Erreur initialisation embeddings: {e}")
            raise

    def _init_llm(self):
        try:
            self.llm = Ollama(
                base_url=self.ollama_base_url,
                model=self.model_name,
                temperature=0.1,
                top_p=0.9,
                num_ctx=4096,
            )
            logger.info(f"✅ LLM {self.model_name} initialisé")
        except Exception as e:
            logger.error(f" Erreur initialisation LLM: {e}")
            raise

    def _init_vectorstore(self):
        try:
            chroma_settings = Settings(
                persist_directory=self.persist_directory, anonymized_telemetry=False
            )
            self.vectorstore = None
            logger.info("✅ ChromaDB initialisé")
        except Exception as e:
            logger.error(f" Erreur initialisation ChromaDB: {e}")
            raise

    def _init_memory(self):
        """
        Initialise la mémoire conversationnelle selon le type choisi.
        memory_type: 'none', 'summary', 'window', 'vectorstore'
        """
        if self.memory_type == "summary":
            # ConversationSummaryMemory (résumé automatique)
            from langchain.chat_models import (
                ChatOpenAI,
            )  # Remplacer par ton LLM local si besoin

            self.memory = ConversationSummaryMemory(
                llm=self.llm,  # Utilise le LLM local
                memory_key="chat_history",
                **self.memory_kwargs,
            )
        elif self.memory_type == "window":
            # ConversationBufferWindowMemory (fenêtre glissante)
            self.memory = ConversationBufferWindowMemory(
                k=self.memory_kwargs.get("k", 3), return_messages=True
            )
        elif self.memory_type == "vectorstore" and FAISS_AVAILABLE:
            # VectorStoreRetrieverMemory (mémoire vectorielle)
            embedding_model = HuggingFaceEmbeddings(
                model_name=self.memory_kwargs.get(
                    "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
                )
            )
            vectorstore = FAISS(embedding_function=embedding_model)
            self.memory = VectorStoreRetrieverMemory(
                retriever=vectorstore.as_retriever(
                    search_kwargs={"k": self.memory_kwargs.get("k", 3)}
                ),
                memory_key="chat_history",
            )
        else:
            self.memory = None  # Pas de mémoire

    def _init_qa_chain(self):
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Tu es un expert en automatisation de tests STBTester pour la plateforme Totalplay.
Ta mission est d'être interactif et pertinent :
- Si la demande de l'utilisateur nécessite la génération d'un script ou de code Python, tu DOIS TOUJOURS présenter le code dans un bloc markdown commençant par ```python et finissant par ``` (même pour un petit extrait ou un script minimal). N'affiche jamais de code en dehors de ce bloc.
- Si la demande de l'utilisateur contient explicitement des mots comme "script", "code", "automatiser", "test", ou décrit un scénario de test, alors génère un script Python STBTester complet et fonctionnel, avec uniquement les imports et fonctions nécessaires, en suivant strictement le scénario ou la demande de l'utilisateur. N'utilise que les éléments du contexte pertinents, ne copie pas tout le contexte.
- Si la demande contient un mot-clé d'action simple (ouvrir, aller à, accéder à, lancer, démarrer, etc.) et une application connue (ex: amazon, youtube, netflix, etc.), génère un script minimal pour réaliser cette action, même si la demande n'est pas très détaillée.
- Si la demande contient seulement "code", "script" ou un mot-clé similaire SANS description d'action ou d'application, réponds poliment que tu as besoin d'un scénario ou d'une demande claire pour générer un script, et NE GÉNÈRE PAS de code.
- Si la demande concerne la documentation, le fonctionnement, le rôle du framework, une question fréquente ou une explication sur une fonction, réponds par une explication textuelle claire et concise, sans générer de code.
- Si la demande est une salutation, une question générale, ou ne correspond à aucun cas connu dans le contexte, réponds poliment par du texte, sans générer de code ni de script.
- N'invente jamais de fonctions, d'importations ou de scénarios qui ne sont pas présents dans le contexte fourni.
- Utilise exactement les fonctions et informations du contexte, et uniquement ce qui est nécessaire pour répondre à la demande.
- Pour les imports dans un script, utilise ce format :
import time
import totalplay
import sc_stbt
import stbt_core as stbt
from sc_stbt import <l'application>  # par exemple: from sc_stbt import amazon etc
Jamais comme ceci :
import time
import sc_stbt
from totalplay.Menu import to_live, to_menu
from totalplay import select_menu_items, select_apps
- Utilise une seule instance objet dans les classes d'applications, par exemple :
amazon = sc_stbt.amazon.Menu()
- Si la question ne correspond à aucun cas connu ou ne peut pas être traitée, indique-le poliment sans générer de code.

CONTEXTE :
{context}

DEMANDE :
{question}
""",
        )

    def load_test_cases_data(self, json_data: Dict[str, Any]):
        documents = []
        for case in json_data.get("test_cases", []):
            content = self._format_test_case_content(case)
            metadata = {
                "id": case.get("id"),
                "title": case.get("title"),
                "application": case.get("application"),
                "category": case.get("category"),
            }
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        if not documents:
            raise ValueError("Aucun document à insérer dans la vectorstore.")
        self.vectorstore = Chroma.from_documents(
            documents=documents, embedding=self.embeddings, persist_directory="db"
        )
        print(f" {len(documents)} cas de test chargés dans la vectorstore.")

    def _format_test_case_content(self, test_case: Dict[str, Any]) -> str:
        content_parts = [
            f"ID: {test_case.get('id', '')}",
            f"Titre: {test_case.get('title', '')}",
            f"Application: {test_case.get('application', '')}",
            f"Catégorie: {test_case.get('category', '')}",
            f"Objectif: {test_case.get('objective', '')}",
            f"Prérequis: {test_case.get('preconditions', '')}",
        ]
        if test_case.get("steps"):
            content_parts.append("\nÉtapes:")
            for step in test_case["steps"]:
                step_text = f"  {step.get('step_number', '')}: {step.get('action', '')}"
                if step.get("expected_result"):
                    step_text += f" → {step['expected_result']}"
                if step.get("code_equivalent"):
                    step_text += f" [{step['code_equivalent']}]"
                content_parts.append(step_text)
        if test_case.get("functions_used"):
            content_parts.append(
                f"\nFonctions utilisées: {', '.join(test_case['functions_used'])}"
            )
        if test_case.get("navigation_path"):
            content_parts.append(
                f"Chemin de navigation: {test_case['navigation_path']}"
            )
        if test_case.get("tags"):
            content_parts.append(f"Tags: {', '.join(test_case['tags'])}")
        return "\n".join(content_parts)

    def generate_script(self, user_query: str) -> Dict[str, Any]:
        if not hasattr(self, "qa_chain"):
            self.setup_qa_chain()
        try:
            similar_tests = self.search_similar_tests(user_query, k=5)
            context = "\n\n".join([doc["content"] for doc in similar_tests])
            result = self.qa_chain({"query": user_query, "context": context})
            script = self._extract_python_script(result["result"])
            script = self.add_missing_imports(script)
            sources = []
            for doc in result.get("source_documents", []):
                sources.append(
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata,
                    }
                )
            return {
                "script": script,
                "raw_response": result["result"],
                "sources": sources,
                "query": user_query,
            }
        except Exception as e:
            logger.error(f"❌ Erreur génération script: {e}")
            raise

    def _extract_python_script(self, response: str) -> str:
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        return response.strip()

    def search_similar_tests(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.vectorstore:
            raise ValueError("Vectorstore non initialisé")
        docs = self.vectorstore.similarity_search(query, k=k)
        results = []
        for doc in docs:
            results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": "High",
                }
            )
        return results

    def setup_qa_chain(self):
        """Configure la chaîne QA avec le retriever"""
        if not self.vectorstore:
            raise ValueError("Vectorstore non initialisé. Chargez d'abord les données.")

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={
                "k": 5,  # Top 5 documents les plus pertinents
                "fetch_k": 10,  # Récupère 10 docs puis sélectionne 5
                "lambda_mult": 0.7,  # Balance diversité/similarité
            },
        )

        from langchain.chains import RetrievalQA

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True,
        )

        logger.info(" Chaîne QA configurée")

    def add_missing_imports(self, script: str) -> str:
        """Ajoute les imports Python nécessaires si manquants dans le script généré."""
        required_imports = [
            "import totalplay",
            "import sc_stbt",
            "import time",
        ]
        lines = script.splitlines()
        existing_imports = set(
            line.strip() for line in lines if line.strip().startswith("import ")
        )
        missing = [imp for imp in required_imports if imp not in existing_imports]
        # Ajoute les imports manquants en haut du script
        if missing:
            script = "\n".join(missing) + "\n" + script
        return script

    def get_file_hash(self, filepath):
        """Calcule le hash SHA256 d'un fichier."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def ensure_vectorstore(self, json_data: Dict[str, Any], data_path: str = None):
        """
        Charge la vectorstore depuis le disque si elle existe et que le dataset n'a pas changé, sinon la (re)génère.
        """
        db_path = "db"
        hash_path = os.path.join(db_path, "data_hash.txt")
        if data_path is None:
            data_path = "data/new_data.json"
        current_hash = self.get_file_hash(data_path)
        previous_hash = None
        if os.path.exists(hash_path):
            with open(hash_path, "r", encoding="utf-8") as f:
                previous_hash = f.read().strip()
        # Si la vectorstore existe ET le hash n'a pas changé, on charge
        if (
            os.path.exists(db_path)
            and os.listdir(db_path)
            and previous_hash == current_hash
        ):
            self.vectorstore = Chroma(
                embedding_function=self.embeddings, persist_directory=db_path
            )
            logger.info("✅ Vectorstore chargée depuis le disque (pas de ré-embedding)")
            # Ne pas recharger FAQ/framework_info pour éviter les doublons
        else:
            self.load_test_cases_data(json_data)
            # Ajoute FAQ et framework_info après les tests
            self.load_faq_and_framework_info(json_data)
            # Sauvegarde le hash courant
            os.makedirs(db_path, exist_ok=True)
            with open(hash_path, "w", encoding="utf-8") as f:
                f.write(current_hash)

    def stream_generate_script(self, user_query: str):
        """
        Génère un script STBTester en streaming (token par token ou bloc par bloc).
        Utilise le mode streaming du LLM Ollama.
        """
        if not hasattr(self, "qa_chain"):
            self.setup_qa_chain()
        try:
            similar_tests = self.search_similar_tests(user_query, k=5)
            context = "\n\n".join([doc["content"] for doc in similar_tests])
            # On utilise le LLM en mode streaming
            # On suppose que self.llm supporte .stream (Ollama le supporte)
            prompt = self.prompt_template.format(context=context, question=user_query)
            stream = self.llm.stream(prompt)
            script = ""
            for chunk in stream:
                script += chunk
                yield chunk  # On yield chaque morceau dès qu'il arrive
        except Exception as e:
            logger.error(f"❌ Erreur génération script (stream): {e}")
            raise

    def load_faq_and_framework_info(self, json_data: Dict[str, Any]):
        """
        Ajoute les Q/R de assistant_faq et les sections de framework_info comme documents dans la vectorstore.
        Cette version évite les plantages sur les valeurs non str/list/dict et gère les cas vides.
        """
        documents = []
        # FAQ
        for faq in json_data.get("assistant_faq", []):
            try:
                content = f"Question: {faq.get('question', '')}\nRéponse: {faq.get('answer', '')}"
                metadata = {"type": "faq"}
                documents.append(Document(page_content=content, metadata=metadata))
            except Exception as e:
                logger.warning(f"FAQ mal formée ignorée: {faq} ({e})")
        # Framework info (on découpe chaque section)
        fw = json_data.get("framework_info", {})
        for key, value in fw.items():
            try:
                if isinstance(value, dict):
                    for subkey, subval in value.items():
                        content = f"{key.capitalize()} - {subkey}: {subval}"
                        metadata = {
                            "type": "framework_info",
                            "section": key,
                            "subsection": subkey,
                        }
                        documents.append(
                            Document(page_content=content, metadata=metadata)
                        )
                elif isinstance(value, list):
                    content = f"{key.capitalize()}:\n" + "\n".join(
                        str(x) for x in value
                    )
                    metadata = {"type": "framework_info", "section": key}
                    documents.append(Document(page_content=content, metadata=metadata))
                elif value is not None:
                    content = f"{key.capitalize()}: {value}"
                    metadata = {"type": "framework_info", "section": key}
                    documents.append(Document(page_content=content, metadata=metadata))
            except Exception as e:
                logger.warning(f"framework_info mal formée ignorée: {key} ({e})")
        # Ajoute à la vectorstore existante ou crée une nouvelle
        if documents:
            if self.vectorstore:
                self.vectorstore.add_documents(documents)
            else:
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory="db",
                )
            print(f"{len(documents)} docs FAQ/framework_info ajoutés à la vectorstore.")
        else:
            print("Aucun document FAQ/framework_info à ajouter.")
