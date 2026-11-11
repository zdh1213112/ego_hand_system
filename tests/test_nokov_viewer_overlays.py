import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from visualize_nokov_mano import overlay_catalog, overlay_media_files


class OverlayMediaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name) / 'video_overlay_aligned'
        self.directory.mkdir()
        self.video = self.directory / 'mano_overlay_cam0_left.mp4'
        self.video.write_bytes(b'encoded fixture')
        self.summary = dict(preview_only=False, frames=123)
        self.validation = dict(encoded_videos=[dict(file=self.video.name, tail_decodable=True,
            bytes=self.video.stat().st_size, nb_frames='123', duration='4.1', width=1920, height=1200)])

    def test_uses_real_file_and_encoding_metadata(self):
        videos = overlay_media_files(self.directory, self.summary, self.validation)
        self.assertEqual(list(videos), ['cam0_left'])
        self.assertEqual(videos['cam0_left']['file'], 'video_overlay_aligned/mano_overlay_cam0_left.mp4')
        self.assertEqual(videos['cam0_left']['duration_s'], 4.1)

    def test_missing_or_incomplete_file_is_unavailable(self):
        self.video.unlink()
        self.assertEqual(overlay_media_files(self.directory, self.summary, self.validation), {})
        self.video.write_bytes(b'partial')
        self.assertEqual(overlay_media_files(self.directory, self.summary, self.validation), {})

    def test_preview_and_failed_decode_do_not_become_available(self):
        self.summary['preview_only'] = True
        self.assertEqual(overlay_media_files(self.directory, self.summary, self.validation), {})
        self.summary['preview_only'] = False
        self.validation['encoded_videos'][0]['tail_decodable'] = False
        self.assertEqual(overlay_media_files(self.directory, self.summary, self.validation), {})

    def test_frame_mismatch_and_unlisted_paths_are_unavailable(self):
        self.validation['encoded_videos'][0]['nb_frames'] = '122'
        self.assertEqual(overlay_media_files(self.directory, self.summary, self.validation), {})
        self.validation['encoded_videos'][0]['nb_frames'] = '123'
        self.validation['encoded_videos'][0]['file'] = '../video_overlay/mano_overlay_cam0_left.mp4'
        self.assertEqual(overlay_media_files(self.directory, self.summary, self.validation), {})

    def write_catalog_metadata(self, alignment):
        for name, value in [('overlay_summary.json', self.summary),
                            ('overlay_validation.json', self.validation),
                            ('alignment_validation.json', alignment)]:
            (self.directory / name).write_text(json.dumps(value))

    def test_optimization_success_cannot_make_aligned_default(self):
        self.write_catalog_metadata(dict(status='candidate_requires_visual_validation', success=True))
        catalog = overlay_catalog(self.directory.parent)
        self.assertEqual(catalog['default'], 'original')
        self.assertIn('cam0_left', catalog['variants']['aligned']['videos'])
        self.assertIn('候选', catalog['variants']['aligned']['label'])

    def test_recalibration_does_not_relabel_old_videos(self):
        self.summary['projection_config_sha256'] = 'old-mount'
        self.validation['projection_config_sha256'] = 'old-mount'
        self.write_catalog_metadata(dict(status='partial', projection_config_sha256='new-mount'))
        self.assertEqual(overlay_catalog(self.directory.parent)['variants']['aligned']['videos'], {})
        self.summary['projection_config_sha256'] = 'new-mount'
        self.validation['projection_config_sha256'] = 'new-mount'
        self.write_catalog_metadata(dict(status='partial', projection_config_sha256='new-mount'))
        self.assertIn('cam0_left', overlay_catalog(self.directory.parent)['variants']['aligned']['videos'])

    def test_default_requires_approval_validated_samples_and_existing_video(self):
        alignment = dict(status='validated_sparse_scope', approved_for_default=True,
                         by_camera={'0': dict(status='validated', validation_frames=[46, 120],
                             after_px=dict(count=12, median=2.1, p95=4.2))})
        self.write_catalog_metadata(alignment)
        self.assertEqual(overlay_catalog(self.directory.parent)['default'], 'aligned')
        alignment['status'] = 'partial'
        alignment['by_camera']['0']['status'] = 'validated_sparse_scope'
        self.write_catalog_metadata(alignment)
        self.assertEqual(overlay_catalog(self.directory.parent)['default'], 'aligned')
        alignment['approved_for_default'] = False
        self.write_catalog_metadata(alignment)
        self.assertEqual(overlay_catalog(self.directory.parent)['default'], 'original')
        alignment['approved_for_default'] = True
        self.write_catalog_metadata(alignment)
        self.video.unlink()
        self.assertEqual(overlay_catalog(self.directory.parent)['default'], 'original')

    def test_failed_or_empty_validation_cannot_make_aligned_default(self):
        for camera in [dict(status='failed'), dict(status='not_validated'),
                       dict(status='validated', validation_frames=[]),
                       dict(status='validated', validation_frames=[46], after_px=dict(count=0)),
                       dict(status='validated', validation_frames=[46], after_px=dict(count=2, median=None, p95=None))]:
            with self.subTest(camera=camera):
                self.write_catalog_metadata(dict(status='validated_sparse_scope',
                    approved_for_default=True, by_camera={'0': camera}))
                self.assertEqual(overlay_catalog(self.directory.parent)['default'], 'original')

    def test_partial_json_does_not_break_viewer_export(self):
        (self.directory / 'overlay_summary.json').write_text('{')
        catalog = overlay_catalog(self.directory.parent)
        self.assertEqual(catalog['variants']['aligned']['videos'], {})


if __name__ == '__main__':
    unittest.main()
