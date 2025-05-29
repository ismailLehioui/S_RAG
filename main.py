import json
import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
import logging


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class STBTesterRAG:
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model_name: str = "qwen3:8b",  # Modèle optimisé pour les tâches de génération de code
        persist_directory: str = "./chroma_db",
    ):
        """
        Initialise le pipeline RAG pour STBTester

        Args:
            ollama_base_url: URL de base d'Ollama
            model_name: Nom du modèle à utiliser
            persist_directory: Répertoire de persistance ChromaDB
        """
        self.ollama_base_url = ollama_base_url
        self.model_name = model_name
        self.persist_directory = persist_directory

        # Initialisation des composants
        self._init_embeddings()
        self._init_llm()
        self._init_vectorstore()
        self._init_qa_chain()

    def _init_embeddings(self):
        """Initialise les embeddings Ollama"""
        try:
            self.embeddings = OllamaEmbeddings(
                base_url=self.ollama_base_url,
                model="nomic-embed-text",  # Modèle d'embedding efficace
            )
            logger.info("✅ Embeddings initialisés")
        except Exception as e:
            logger.error(f" Erreur initialisation embeddings: {e}")
            raise

    def _init_llm(self):
        """Initialise le LLM Ollama"""
        try:
            self.llm = Ollama(
                base_url=self.ollama_base_url,
                model=self.model_name,
                temperature=0.1,  # Température basse pour plus de consistance
                top_p=0.9,
                num_ctx=4096,  # Contexte large pour les scripts longs
            )
            logger.info(f"✅ LLM {self.model_name} initialisé")
        except Exception as e:
            logger.error(f" Erreur initialisation LLM: {e}")
            raise

    def _init_vectorstore(self):
        """Initialise ChromaDB"""
        try:
            # Configuration ChromaDB
            chroma_settings = Settings(
                persist_directory=self.persist_directory, anonymized_telemetry=False
            )

            self.vectorstore = None
            logger.info("✅ ChromaDB initialisé")
        except Exception as e:
            logger.error(f" Erreur initialisation ChromaDB: {e}")
            raise

    def _init_qa_chain(self):
        """Initialise la chaîne QA"""
        # Template de prompt spécialisé pour STBTester
        # 5. Ajoute des commentaires explicatifs
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
        """
        Charge les cas de test depuis un dictionnaire JSON et les insère dans la vectorstore
        """
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

        # Création de la vectorstore avec les documents
        self.vectorstore = Chroma.from_documents(
            documents=documents, embedding=self.embeddings, persist_directory="db"
        )

        print(f" {len(documents)} cas de test chargés dans la vectorstore.")

    def _prepare_documents(self, json_data: Dict[str, Any]) -> List[Document]:
        """
        Prépare les documents à partir des données JSON

        Args:
            json_data: Données JSON des cas de test

        Returns:
            Liste de documents LangChain
        """
        documents = []

        # Document général avec informations du dataset
        dataset_info = json_data.get("dataset_info", {})
        dataset_doc = Document(
            page_content=f"""
Dataset: {dataset_info.get('name', '')}
Description: {dataset_info.get('description', '')}
Applications: {', '.join(dataset_info.get('applications', []))}
Catégories: {', '.join(dataset_info.get('categories', []))}
            """.strip(),
            metadata={"type": "dataset_info"},
        )
        documents.append(dataset_doc)

        # Documents pour chaque cas de test
        for test_case in json_data.get("test_cases", []):
            # Contenu principal du cas de test
            content = self._format_test_case_content(test_case)

            # Métadonnées enrichies
            metadata = {
                "id": test_case.get("id", ""),
                "title": test_case.get("title", ""),
                "category": test_case.get("category", ""),
                "application": test_case.get("application", ""),
                "complexity": test_case.get("complexity", ""),
                "type": "test_case",
            }

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

            # Document séparé pour le code
            if test_case.get("code_snippet"):
                code_content = f"""
Test Case: {test_case.get('title', '')}
Application: {test_case.get('application', '')}
Category: {test_case.get('category', '')}

CODE SNIPPET:
{test_case['code_snippet']}

FUNCTIONS USED:
{', '.join(test_case.get('functions_used', []))}

PATTERNS:
{', '.join(test_case.get('patterns', []))}
                """.strip()

                code_doc = Document(
                    page_content=code_content,
                    metadata={**metadata, "type": "code_snippet"},
                )
                documents.append(code_doc)

        return documents

    def _format_test_case_content(self, test_case: Dict[str, Any]) -> str:
        """Formate le contenu d'un cas de test"""
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

        # Étapes
        if test_case.get("steps"):
            content_parts.append("\nÉtapes:")
            for step in test_case["steps"]:
                step_text = f"  {step.get('step_number', '')}: {step.get('action', '')}"
                if step.get("expected_result"):
                    step_text += f" → {step['expected_result']}"
                if step.get("code_equivalent"):
                    step_text += f" [{step['code_equivalent']}]"
                content_parts.append(step_text)

        # Informations techniques
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

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True,
        )

        logger.info(" Chaîne QA configurée")

    def generate_script(self, user_query: str) -> Dict[str, Any]:
        """
        Génère un script STBTester basé sur la requête utilisateur, avec contexte enrichi

        Args:
            user_query: Description du test souhaité

        Returns:
            Dictionnaire avec le script généré et les sources
        """
        if not hasattr(self, "qa_chain"):
            self.setup_qa_chain()

        try:
            # Récupère plusieurs cas de test similaires pour enrichir le contexte
            similar_tests = self.search_similar_tests(user_query, k=5)
            context = "\n\n".join([doc["content"] for doc in similar_tests])
            # Exécution de la requête avec contexte enrichi
            result = self.qa_chain({"query": user_query, "context": context})

            # Extraction du script Python du résultat
            script = self._extract_python_script(result["result"])

            # Informations sur les sources utilisées
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
        """Extrait le code Python de la réponse"""
        # Cherche le code entre ```python et ```
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()

        # Si pas de markdown, retourne la réponse complète
        return response.strip()

    def search_similar_tests(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Recherche des tests similaires

        Args:
            query: Requête de recherche
            k: Nombre de résultats

        Returns:
            Liste des tests similaires
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore non initialisé")

        docs = self.vectorstore.similarity_search(query, k=k)
        results = []

        for doc in docs:
            results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": "High",  # ChromaDB ne retourne pas de score par défaut
                }
            )

        return results


def main():
    """Fonction principale de démonstration pour RAG avec data_youtube_all.json et Ollama local (qwen3:8b)"""
    import json
    import os

    # Chemin vers le fichier de dataset fusionné
    data_path = os.path.join("data", "data_youtube_all.json")
    output_dir = os.path.join("generated_tests", "youtube")
    os.makedirs(output_dir, exist_ok=True)

    # Chargement du fichier JSON
    with open(data_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    try:
        # Initialisation du pipeline RAG
        print(" Initialisation du pipeline RAG STBTester...")
        rag = STBTesterRAG()

        # Chargement des données
        print(" Chargement des cas de test...")
        rag.load_test_cases_data(json_data)

        # Configuration de la chaîne QA
        print(" Configuration de la chaîne QA...")
        rag.setup_qa_chain()

        # Exemples de scénarios mixtes à tester
        scenarios = [
            "Lancer netflix depuis le menu STREAMING, naviguer à droite deux fois, attendre 10 secondes, puis quitter proprement l’application."
            # "Ouvrir YouTube, lancer une vidéo, puis toutes les 5 minutes naviguer vers le bas et attendre 30 minutes au total avant de quitter.",
            # "Démarrer YouTube, attendre le chargement, vérifier qu’il n’y a pas d’écran noir pendant 1 heure, puis sortir de l’application.",
            # "Aller sur YouTube via STREAMING, naviguer à droite et en bas, attendre 20 secondes, puis revenir au menu principal.",
            # "Lancer YouTube, naviguer dans plusieurs directions (haut, bas, droite, gauche) en boucle pendant 10 minutes, puis quitter."
        ]

        for i, query in enumerate(scenarios, 1):
            print(f"\n🎯 Génération d'un script de test pour le scénario mixte {i}...")
            result = rag.generate_script(query)
            print(f"\nScénario: {query}")
            print("=" * 50)
            print(result["script"])
            print("=" * 50)
            # Sauvegarde du script généré dans un fichier dédié
            script_filename = os.path.join(output_dir, f"test_youtube_scenario_{i}.py")
            with open(script_filename, "w", encoding="utf-8") as f_script:
                f_script.write(result["script"])
            print(f"\n✅ Script sauvegardé dans : {script_filename}")
            print(f"\n📚 Sources utilisées: {len(result['sources'])}")
            for j, source in enumerate(result["sources"][:2]):
                print(f"  {j+1}. {source['metadata'].get('title', 'N/A')}")

    except Exception as e:
        print(f"Erreur: {e}")


if __name__ == "__main__":
    main()
