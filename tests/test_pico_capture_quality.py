import csv
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_pico_capture_quality import Thresholds, analyze_capture, main, missing_ranges


class PicoCaptureQualityTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def capture(self, name="capture", count=8, serials=("SERIAL_A", "SERIAL_B")):
        directory = self.root / name / "pico"
        directory.mkdir(parents=True)
        metadata = {
            "schema": "pico_raw_tracking_capture_v1",
            "status": "complete",
            "poll_hz": 180,
            "selected_tracker_serials": list(serials),
            "observed_tracker_serials": list(serials),
            "rows": {
                "head": count,
                "tracker": count * len(serials),
                "motion_packets": count,
            },
            "duplicate_or_non_increasing_polls_not_written": {"head": 25, "motion": 25},
        }
        head, trackers = [], []
        for i in range(count):
            common = {
                "sample_index": i,
                "host_receive_perf_ns": 10**16 + i * 13_888_889,
                "host_receive_unix_ns": 1793876377000000000 + i * 13_888_889,
                "valid_numeric": 1,
                "x_m": 0,
                "y_m": 0,
                "z_m": 0,
                "qx": 0,
                "qy": 0,
                "qz": 0,
                "qw": 1,
            }
            timestamp = 4355067032740 + i * 13_888_889
            head.append({**common, "head_timestamp_ns": timestamp})
            for j, serial in enumerate(serials):
                trackers.append(
                    {
                        **common,
                        "motion_timestamp_ns": timestamp,
                        "tracker_index": j,
                        "tracker_serial": serial,
                        "selected": 1,
                        "vx_m_s": 0,
                        "vy_m_s": 0,
                        "vz_m_s": 0,
                    }
                )
        self.write(directory, metadata, head, trackers)
        return directory, metadata, head, trackers

    def write(self, directory, metadata, head, trackers):
        (directory / "capture_metadata.json").write_text(json.dumps(metadata))
        for name, rows in (
            ("pico_head_poses.csv", head),
            ("pico_tracker_poses.csv", trackers),
        ):
            fields = list(rows[0]) if rows else list(head[0])
            with (directory / name).open(
                "w", newline="", encoding="utf-8-sig"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    def test_actual_device_rate_and_zero_derivatives_do_not_use_poll_rate(self):
        directory, _, _, _ = self.capture()
        report, _, _ = analyze_capture(directory.parent)
        self.assertEqual(report["status"], "ok")
        self.assertAlmostEqual(report["streams"][0]["average_hz"], 72, places=4)
        self.assertEqual(report["poll_hz_metadata"], 180)
        self.assertFalse(report["absolute_accuracy_evaluated"])
        self.assertTrue(
            report["streams"][1]["auxiliary_fields"]["vx_m_s"]["all_finite_values_zero"]
        )
        self.assertEqual(
            report["streams"][0]["tracking_state_assessment"],
            "not_available_in_v1_pose_format",
        )
        self.assertTrue(report["raw_inputs_unchanged"])

    def test_sign_flip_of_quaternion_is_not_a_jump(self):
        directory, meta, head, trackers = self.capture()
        for i, row in enumerate(head):
            row["qw"] = (-1) ** i
        self.write(directory, meta, head, trackers)
        report, _, _ = analyze_capture(directory)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["streams"][0]["rotation_step_deg"]["max"], 0)

    def test_invalid_values_break_motion_edges_even_when_flag_is_one(self):
        directory, meta, head, trackers = self.capture()
        head[2]["x_m"] = "nan"
        head[3]["x_m"] = 50
        head[4]["qw"] = 0
        head[5]["qw"] = 2
        self.write(directory, meta, head, trackers)
        report, samples, _ = analyze_capture(directory)
        self.assertEqual(report["issue_counts"]["invalid_pose"], 3)
        self.assertEqual(report["streams"][0]["numeric_valid_rows"], 5)
        self.assertIsNone(samples["head"][3].step)
        self.assertIsNone(samples["head"][4].step)
        self.assertIsNone(samples["head"][6].step)

    def test_real_position_and_rotation_spikes_are_retained(self):
        directory, meta, head, trackers = self.capture()
        head[3]["x_m"] = 0.2
        head[3]["qz"], head[3]["qw"] = (
            math.sin(math.radians(10)),
            math.cos(math.radians(10)),
        )
        self.write(directory, meta, head, trackers)
        report, samples, _ = analyze_capture(directory)
        self.assertEqual(report["issue_counts"]["position_step"], 2)
        self.assertEqual(report["issue_counts"]["rotation_step"], 2)
        self.assertAlmostEqual(samples["head"][3].step, 200)
        self.assertAlmostEqual(samples["head"][3].angle, 20)
        self.assertEqual(report["streams"][0]["review_samples"], 2)

    def test_device_time_rollback_is_not_sorted_or_bridged(self):
        directory, meta, head, trackers = self.capture()
        head[3]["head_timestamp_ns"] = head[2]["head_timestamp_ns"] - 1
        head[3]["x_m"] = 0.2
        self.write(directory, meta, head, trackers)
        report, samples, _ = analyze_capture(directory)
        self.assertEqual(report["issue_counts"]["non_increasing_device_time"], 1)
        self.assertEqual([s.index for s in samples["head"]], list(range(8)))
        self.assertIsNone(samples["head"][3].step)
        self.assertIsNone(samples["head"][4].step)
        self.assertIsNone(report["streams"][0]["average_hz"])

    def test_device_gap_is_distinct_from_host_scheduling_jitter(self):
        directory, meta, head, trackers = self.capture()
        for row in head[4:]:
            row["head_timestamp_ns"] += 100_000_000
            row["x_m"] = 10
        self.write(directory, meta, head, trackers)
        report, samples, _ = analyze_capture(directory)
        self.assertEqual(report["issue_counts"]["device_time_gap"], 1)
        self.assertIsNone(samples["head"][4].step)
        for i, row in enumerate(head):
            row["head_timestamp_ns"] = 4355067032740 + i * 13_888_889
            row["x_m"] = 0
            row["host_receive_perf_ns"] = 10**16 + i * 100_000_000
        self.write(directory, meta, head, trackers)
        report, _, _ = analyze_capture(directory)
        self.assertEqual(report["status"], "ok")
        self.assertAlmostEqual(report["streams"][0]["average_hz"], 72, places=4)

    def test_missing_tracker_placeholder_and_absent_packet_are_reported(self):
        directory, meta, head, trackers = self.capture()
        missing = next(
            r
            for r in trackers
            if r["sample_index"] == 2 and r["tracker_serial"] == "SERIAL_A"
        )
        missing["valid_numeric"] = 0
        missing["tracker_index"] = -1
        for key in ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"):
            missing[key] = ""
        trackers[:] = [
            r
            for r in trackers
            if not (r["sample_index"] == 5 and r["tracker_serial"] == "SERIAL_B")
        ]
        meta["rows"]["tracker"] = len(trackers)
        self.write(directory, meta, head, trackers)
        report, samples, _ = analyze_capture(directory)
        self.assertEqual(report["issue_counts"]["missing_tracker_packets"], 1)
        self.assertEqual(report["issue_counts"]["numeric_flag_invalid"], 1)
        self.assertIsNone(samples["tracker:SERIAL_A"][3].step)
        self.assertIsNone(samples["tracker:SERIAL_B"][5].step)

    def test_tracker_disappearing_entirely_and_reordered_tracker_indices(self):
        directory, meta, head, trackers = self.capture()
        for row in trackers:
            row["tracker_index"] = (row["tracker_index"] + row["sample_index"]) % 2
        self.write(directory, meta, head, trackers)
        self.assertEqual(analyze_capture(directory)[0]["status"], "ok")
        trackers[:] = [r for r in trackers if r["tracker_serial"] == "SERIAL_A"]
        meta["rows"]["tracker"] = len(trackers)
        self.write(directory, meta, head, trackers)
        report, _, _ = analyze_capture(directory)
        self.assertIn("missing_tracker_packets", report["issue_counts"])
        self.assertIn("empty_stream", report["issue_counts"])
        self.assertEqual(len(report["streams"]), 3)

    def test_duplicate_indices_and_conflicting_motion_packet_timestamps(self):
        directory, meta, head, trackers = self.capture()
        trackers[4]["sample_index"] = 1
        trackers[7]["motion_timestamp_ns"] += 100
        self.write(directory, meta, head, trackers)
        report, _, _ = analyze_capture(directory)
        self.assertIn("non_increasing_sample_index", report["issue_counts"])
        self.assertIn("packet_timestamp_conflict", report["issue_counts"])

    def test_missing_header_and_metadata_count_mismatch_cannot_pass(self):
        directory, _, _, _ = self.capture()
        (directory / "pico_head_poses.csv").write_text("sample_index\n0\n")
        report, _, _ = analyze_capture(directory)
        self.assertIn("invalid_columns", report["issue_counts"])
        self.assertIn("metadata_row_mismatch", report["issue_counts"])
        self.assertEqual(report["status"], "needs_review")

    def test_nonzero_initial_index_and_incomplete_metadata_are_reported(self):
        directory, meta, head, trackers = self.capture()
        for row in head:
            row["sample_index"] += 10
        meta["status"] = "interrupted"
        self.write(directory, meta, head, trackers)
        report, _, _ = analyze_capture(directory)
        self.assertIn("nonzero_first_sample_index", report["issue_counts"])
        self.assertIn("capture_not_complete", report["issue_counts"])

    def test_one_sample_cannot_certify_continuity(self):
        directory, _, _, _ = self.capture(count=1)
        report, _, _ = analyze_capture(directory)
        self.assertEqual(report["status"], "needs_review")
        self.assertEqual(report["issue_counts"]["insufficient_stream_samples"], 3)

    def test_host_clock_rollback_and_missing_metadata_are_visible(self):
        directory, meta, head, trackers = self.capture()
        head[3]["host_receive_unix_ns"] = head[2]["host_receive_unix_ns"] - 1
        self.write(directory, meta, head, trackers)
        (directory / "capture_metadata.json").unlink()
        report, _, _ = analyze_capture(directory)
        self.assertIn("missing_metadata", report["issue_counts"])
        self.assertIn("non_increasing_host_unix", report["issue_counts"])

    def test_metadata_selection_all_and_legacy_poll_counter(self):
        directory, meta, head, trackers = self.capture()
        meta["selected_tracker_serials"] = "all"
        meta["duplicate_polls_not_written"] = meta.pop(
            "duplicate_or_non_increasing_polls_not_written"
        )
        self.write(directory, meta, head, trackers)
        report, _, _ = analyze_capture(directory)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["duplicate_polls_not_written"]["head"], 25)

    def test_blank_or_malformed_rows_cannot_disappear(self):
        directory, _, _, _ = self.capture()
        with (directory / "pico_head_poses.csv").open("a") as stream:
            stream.write("8,123\n")
        report, samples, _ = analyze_capture(directory)
        self.assertIn("malformed_csv_row", report["issue_counts"])
        self.assertEqual(len(samples["head"]), 9)
        self.assertTrue(samples["head"][-1].reasons)

    def test_explicit_expected_hz_controls_only_device_gap_threshold(self):
        directory, _, _, _ = self.capture()
        report, _, _ = analyze_capture(directory, Thresholds(expected_hz=1000))
        self.assertIn("device_time_gap", report["issue_counts"])
        self.assertAlmostEqual(report["streams"][0]["average_hz"], 72, places=4)
        self.assertEqual(
            report["streams"][0]["gap_period_source"], "explicit_expected_hz"
        )

    def test_missing_ranges_handles_large_counts_without_expanding(self):
        self.assertEqual(
            missing_ranges([0, 1, 5, 5], 10**12),
            (10**12 - 3, [[2, 4], [6, 10**12 - 1]]),
        )

    def test_batch_cli_escaping_provenance_and_exit_status(self):
        first, meta, head, trackers = self.capture("one", serials=("A<script>",))
        second, _, _, _ = self.capture("two")
        fingerprints = {
            p: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in self.root.rglob("*.csv")
        }
        output = self.root / "reports"
        self.assertEqual(
            main([str(first.parent), str(second), "--output", str(output)]), 0
        )
        document = (output / "one/report.html").read_text()
        self.assertIn("PICO 自身原始位姿", document)
        self.assertIn("A&lt;script&gt;", document)
        self.assertNotIn("A<script>", document)
        self.assertTrue((output / "two/issues.csv").is_file())
        self.assertTrue((output / "index.html").is_file())
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(len(summary), 2)
        for path, digest in fingerprints.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        head[3]["x_m"] = 1
        self.write(first, meta, head, trackers)
        self.assertEqual(main([str(first), "--output", str(output)]), 1)

    def test_output_under_input_and_missing_input_return_error(self):
        directory, _, _, _ = self.capture()
        self.assertEqual(
            main([str(directory), "--output", str(directory.parent / "reports")]), 2
        )
        self.assertEqual(
            main([str(directory), "--output", str(directory.parent.parent)]), 2
        )
        self.assertEqual(
            main([str(self.root / "absent"), "--output", str(self.root / "reports")]), 2
        )


if __name__ == "__main__":
    unittest.main()
