from app_for_toolboxes.rag.data_loader import load_orange_toolbox_dataset
from app_for_toolboxes.rag.utils import search_functions

if __name__ == "__main__":
    dataset = load_orange_toolbox_dataset()
    print("Dataset Orange toolboxes chargé. Nombre de fonctions:", len(dataset))

    # Exemple d'interrogation simple
    query = input("Entrez un mot-clé ou nom de fonction à rechercher : ")
    results = search_functions(dataset, query)
    print(f"{len(results)} résultat(s) trouvé(s) :")
    for entry in results:
        print(
            f"\nFonction : {entry['function_name']}\nDescription : {entry['description']}\nSignature : {entry['signature']}\nExemple : {entry['usage_examples'][0] if entry['usage_examples'] else 'N/A'}\n"
        )
