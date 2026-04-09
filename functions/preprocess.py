import spikeinterface.preprocessing as spre


def preprocess_recordings(sess):
    """
    Applique un preprocessing standard sur tous les recordings de la session.
    Remplace sess["recordings"] par les recordings preprocessés.
    """

    print(f"\n=== PREPROCESS: {sess['session_name']} ===")

    recordings = sess["recordings"]
    recordings_pp = {}

    for probe_name, rec in recordings.items():
        print(f"\n[Probe: {probe_name}]")

        # 1) Bandpass
        rec_pp = spre.bandpass_filter(rec, freq_min=300, freq_max=6000)

        # 2) Bad channels (méthode IBL-like)
        bad_channel_ids, _ = spre.detect_bad_channels(
            rec_pp,
            method="coherence+psd"
        )

        print(f"  Bad channels: {len(bad_channel_ids)}")

        # 3) Interpolation
        if len(bad_channel_ids) > 0:
            rec_pp = spre.interpolate_bad_channels(
                rec_pp,
                bad_channel_ids=bad_channel_ids
            )
            print("  → interpolated")

        # 4) Common reference
        rec_pp = spre.common_reference(
            rec_pp,
            operator="median",
            reference="global"
        )

        recordings_pp[probe_name] = rec_pp

    # 👉 on remplace directement
    sess["recordings"] = recordings_pp

    return sess