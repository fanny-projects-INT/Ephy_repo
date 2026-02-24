from pathlib import Path
import matplotlib.pyplot as plt

from functions.dashboard import plot_good_spikes_heatmap_with_regions

alf_probe = Path(r"E:\Aurelien\Data_Mice\VF071_2025_12_18\alf\probe00")
channel_locations = alf_probe / "channel_locations.json"

fig, axes = plot_good_spikes_heatmap_with_regions(
    alf_probe=alf_probe,
    channel_locations_json=channel_locations,
    title="VF071_2025_12_18 • probe00",
    depth_max=4000,          # 0 = tip, 4000 = surface
    heat_time_unit="min",
)

plt.show()
