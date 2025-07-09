import re
from typing import List, Dict


def search_functions(dataset: List[Dict], query: str) -> List[Dict]:
    """Recherche des fonctions dans le dataset dont le nom ou la description correspond à la requête."""
    query = query.lower()
    results = []
    for entry in dataset:
        if (
            query in entry["function_name"].lower()
            or query in entry["description"].lower()
        ):
            results.append(entry)
    return results
