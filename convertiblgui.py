from pathlib import Path
import shutil

import spikeinterface.full as si
from spikeinterface.exporters import export_to_ibl_gui

# =========================================================
# CONFIG
# =========================================================
session_root = Path(r"F:\Data_Mice_IBL\VF074 - copie\2026_03_24")
probe = "probe00"

rec_folder = session_root / "Rec" / probe
ks_folder = session_root / "KS" / probe
alf_folder = session_root / "alf" / probe

n_jobs = 1
chunk_duration = "1s"

# =========================================================
# CHECKS
# =========================================================
if not rec_folder.exists():
    raise FileNotFoundError(f"rec_folder introuvable: {rec_folder}")

sorter_output = ks_folder / "sorter_output"
ks_input = sorter_output if sorter_output.exists() else ks_folder

if not ks_input.exists():
    raise FileNotFoundError(f"ks_folder introuvable: {ks_input}")

print("=== PATHS ===")
print("rec_folder :", rec_folder)
print("ks_input   :", ks_input)
print("alf_folder :", alf_folder)

# =========================================================
# CLEAN OUTPUT FOLDER (CRUCIAL)
# =========================================================
if alf_folder.exists():
    print("\n=== DELETE EXISTING alf/probe00 ===")
    shutil.rmtree(alf_folder)

# =========================================================
# CHECK FILES
# =========================================================
ap_cbin = list(rec_folder.glob("*.ap.cbin"))
print("\n=== FICHIERS ===")
print("AP:", [p.name for p in ap_cbin])

if len(ap_cbin) == 0:
    raise FileNotFoundError("Aucun fichier AP trouvé")

# =========================================================
# LOAD AP
# =========================================================
print("\n=== LOAD AP RECORDING ===")
recording_ap = si.read_cbin_ibl(rec_folder, stream_name="ap")

# =========================================================
# LOAD KS
# =========================================================
print("\n=== LOAD KS OUTPUT ===")
sorting = si.read_kilosort(ks_input)

# =========================================================
# ANALYZER
# =========================================================
print("\n=== BUILD ANALYZER ===")
analyzer = si.create_sorting_analyzer(
    sorting=sorting,
    recording=recording_ap,
    format="memory",
)

# =========================================================
# COMPUTE
# =========================================================
print("\n=== COMPUTE ===")
analyzer.compute(
    [
        "random_spikes",
        "templates",
        "spike_amplitudes",
        "spike_locations",
        "noise_levels",
        "quality_metrics",
    ],
    n_jobs=n_jobs,
    chunk_duration=chunk_duration,
    progress_bar=True,
)

# =========================================================
# EXPORT (SANS LFP)
# =========================================================
print("\n=== EXPORT IBL GUI ===")
export_to_ibl_gui(
    sorting_analyzer=analyzer,
    output_folder=alf_folder,
    lfp_recording=None,  # 🔥 IMPORTANT
    remove_if_exists=True,
    verbose=True,
    n_jobs=n_jobs,
    chunk_duration=chunk_duration,
    progress_bar=True,
)

print("\n✅ DONE")