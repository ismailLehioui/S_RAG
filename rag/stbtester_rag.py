import re
import logging
from typing import List, Dict, Any
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class STBTesterRAG:
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model_name: str = "qwen3:8b",
        persist_directory: str = "./chroma_db",
    ):
        self.ollama_base_url = ollama_base_url
        self.model_name = model_name
        self.persist_directory = persist_directory
        self._init_embeddings()
        self._init_llm()
        self._init_vectorstore()
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

    def _init_qa_chain(self):
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Tu es un expert en automatisation de tests STBTester pour la plateforme Totalplay.
Utilise UNIQUEMENT les informations du contexte fourni pour générer un script Python STBTester.
CONTEXTE:
{context}
DEMANDE:
{question}
INSTRUCTIONS:
1. Génère un script Python complet et fonctionnel
2. Utilise EXACTEMENT les mêmes fonctions que dans les exemples du contexte
3. Respecte la structure et les patterns des scripts existants
4. Inclus tous les imports nécessaires 
5. Assure-toi que le script soit prêt à l'exécution
SCRIPT PYTHON STBTester:
```python""",
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
            f"Complexité: {test_case.get('complexity', '')}",
            f"Durée estimée: {test_case.get('estimated_duration', '')}",
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

    def search_similar_tests(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
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
            "import stbt",
            "import totalplay",
            "import sc_stbt",
            "import time"
        ]
        lines = script.splitlines()
        existing_imports = set(line.strip() for line in lines if line.strip().startswith("import "))
        missing = [imp for imp in required_imports if imp not in existing_imports]
        # Ajoute les imports manquants en haut du script
        if missing:
            script = "\n".join(missing) + "\n" + script
        return script
