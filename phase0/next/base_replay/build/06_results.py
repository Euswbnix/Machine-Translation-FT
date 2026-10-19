"""Collect every step's output into results.json (the one file to read)."""
import hashlib, json, os
B = os.path.expanduser("~/replay_check/build")
J = lambda f: json.load(open(f"{B}/{f}"))
v1, s201, s226, o3, e4 = J("01_validate.json"), J("02_simulate_np201.json"), J("02_simulate_np226.json"), J("03_order_tests.json"), J("04_exposure.json")
b4, c4, d4, f4, s5, s5b = J("04b_epoch_boundary_check.json"), J("04c_p2_order_check.json"), J("04d_toks_test.json"), J("04f_toks_rescan.json"), J("05_sets.json"), J("05b_dup_free.json")
e4r = J("04e_toks_by_range.json")
sha = lambda f: hashlib.sha256(open(f"{B}/{f}", "rb").read()).hexdigest()
res = {
 "what": "v1.1 en-fr base (checkpoints/base_enfr_v1_redo): rows of P1's first 22,294 batches re-trained by the resumed session, per-line exposure over sessions 1+2, and their balance across the E0.5/E0.6 sets",
 "code_of_record": "bde629a (swanlab git_info of both April-24 runs); box copies of src/data/dataset.py and src/training/trainer.py byte-identical, mtimes 2026-04-19/21",
 "environment_april": "python 3.10.18, numpy 2.0.1 (conda numpy-base py310hb5e798b_1), sentencepiece 0.2.1, swanlab 0.7.15, Ryzen 9 9950X3D (AVX-512), host euswbnix (= this box)",
 "step1_2_validation": {k: v1[k] for k in ("kept_rows", "cache_rows", "dropped_over_max_tokens", "skipped_empty_after_strip",
                        "src_lengths_equal_all_rows", "tgt_lengths_equal_all_rows", "src_tokens_equal", "tgt_tokens_equal", "validated", "seconds_total")},
 "step3_composition": {"B_per_epoch_numpy201": [e["B"] for e in s201["epochs"]], "B_per_epoch_numpy226": [e["B"] for e in s226["epochs"]],
                       "numpy201_equals_numpy226_all_epochs": all(a["sorted_order_sha256"] == b["sorted_order_sha256"] for a, b in zip(s201["epochs"], s226["epochs"])),
                       "log": "End of epoch 1 | Step 49,174; End of epoch 2 | Step 98,351 (swanlab console backup, session 1)",
                       "trainer_printed_batches_per_epoch": s201["trainer_len_sampler"],
                       "audit_figure_48502": "= 9,312,233 corpus lines // 192 + 1; the trainer's own figure is 9,309,972 // 192 + 1 = 48,490 (its __len__, not the true count)"},
 "step3_order": {"python_random_words_before_P1_shuffle": 208, "derivation": "swanlab 0.7.15 hardware collectors: NVIDIA 6 + AMD 5 + disk 1 + network 1 random_index() x 16 words; _monitor = 6 in both runs' metadata",
                 "K207_identical_permutation": e4["perm_K207_equals_K208"], "verbatim_sampler_equal": e4["verbatim_equal_per_epoch"],
                 "tests": {"throughput_P1_session1_excl_20k_30k": f4["pearson"], "throughput_session2": d4["S2_K208_E"],
                           "throughput_P2_x0": d4["P2_x0_E"], "throughput_P3_x0": d4["P3_x0_E"], "throughput_by_range": e4r,
                           "loss_window_session1": o3["summary_K_pred"]["T1_session1"], "loss_window_session2": o3["summary_K_pred"]["T1_session2"],
                           "spike_batches_percentile_of_mean_surprisal": o3["summary_K_pred"]["T2_spike_percentiles_at_K_pred"],
                           "loss_window_P2_single_feature": b4["P2_extra_words"], "loss_window_P2_fitted": {k: c4[k] for k in ("P2_r_x0", "rank_of_x0", "best_x", "best_r")}}},
 "step4_replay": {k: e4[k] for k in ("replayed_rows", "replayed_share_of_rows", "p3_prefix_rows", "rows_by_exposure", "replayed_and_p3_prefix_rows",
                                     "session1_spike_groups_steps", "session1_rows_grad_dropped", "session2_unstepped_positions", "session2_rows_unstepped_at_interrupt")},
 "step5_sets": s5["per_set"], "step5_sets_duplicate_free": s5b["per_set"], "rows_pkl_check": s5["rows_pkl_check"], "set_overlaps_pairs": s5["set_overlaps_pairs"],
 "step6_contrasts": s5["contrasts"], "step6_contrasts_duplicate_free": s5b["contrasts"],
 "files_sha256": {f: sha(f) for f in ("replayed_line_mask.npy", "exposure_count_by_line.npy", "p3prefix_line_mask.npy", "p1_position_by_line.npy",
                                      "replayed_row_mask.npy", "index_to_line.npy", "line_status.npy", "comp_np201.npz", "set_occurrences.npz")},
}
json.dump(res, open(f"{B}/results.json", "w"), indent=1)
print("results.json written", os.path.getsize(f"{B}/results.json"), "bytes")
