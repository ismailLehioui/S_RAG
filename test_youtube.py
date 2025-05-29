import totalplay
import sc_stbt
import time


def test_youtube():
    # Ouvrir le menu principal et vérifier qu'il s'affiche
    if not totalplay.Menu.to_menu():
        raise Exception("Impossible d'ouvrir le menu principal")

    # Naviguer vers STREAMING et sélectionner YouTube
    if not (
        totalplay.select_menu_items(menu_item="STREAMING")
        and totalplay.select_apps(name_app="YouTube")
    ):
        raise Exception("Impossible de naviguer vers YouTube")

    # Attendre que YouTube démarre
    time.sleep(
        10
    )  # Vous pouvez ajuster cette valeur en fonction de la vitesse de votre système

    # Vérifier que YouTube est opérationnel
    if not sc_stbt.press_and_wait(
        "KEY_MENU", timeout_secs=6, stable_secs=2
    ):  # Appuyer sur le bouton Menu pour vérifier l'ouverture de YouTube
        raise Exception("YouTube ne semble pas s'ouvrir correctement")

    # Sortir de YouTube et retourner au menu principal
    if not sc_stbt.press_and_wait(
        "KEY_MENU", timeout_secs=6, stable_secs=2
    ):  # Appuyer sur le bouton Menu pour sortir de YouTube
        raise Exception("Impossible de sortir de YouTube")

    print("Le test d'ouverture et de fermeture de YouTube a réussi.")
