from pathlib import Path
import spikeglx


def compress_neuropixels_folder(folder_path, keep_original=True):
    """
    Compresse tous les fichiers .ap.bin et .lf.bin dans un dossier (recursif).
    
    Args:
        folder_path (str or Path): chemin vers le dossier
        keep_original (bool): si False, supprime les .bin après compression
    """
    folder = Path(folder_path)

    # Cherche tous les fichiers .bin pertinents
    bin_files = list(folder.rglob("*.ap.bin")) + list(folder.rglob("*.lf.bin"))

    if len(bin_files) == 0:
        print("❌ Aucun fichier .ap.bin ou .lf.bin trouvé.")
        return

    print(f"🔍 {len(bin_files)} fichiers trouvés")

    for f in bin_files:
        try:
            print(f"\n🚀 Compression : {f}")

            sr = spikeglx.Reader(f)
            sr.compress_file(keep_original=keep_original)

            print("✅ OK")

        except Exception as e:
            print(f"❌ Erreur sur {f} : {e}")


# =========================
# UTILISATION
# =========================

if __name__ == "__main__":
    folder = r"F:\Data_Mice_IBL\VF065\2025_12_12\Rec\probe00"

    compress_neuropixels_folder(
        folder_path=folder,
        keep_original=True   # ⚠️ mets False si tu veux supprimer les .bin
    )