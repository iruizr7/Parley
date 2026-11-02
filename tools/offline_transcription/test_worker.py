"""Fast regression tests; no model downloads or ML dependencies required."""
import copy
import unittest
from worker import annotate, clean_seconds, overlaps, speaker_for,append_recognition_segment


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.turns = [dict(start=0., end=10., speaker='Speaker 1'),
                      dict(start=4., end=5., speaker='Speaker 2')]

    def test_interjection_keeps_both_active_speakers(self):
        self.assertEqual(overlaps(self.turns), [dict(start=4., end=5., speakers=['Speaker 1', 'Speaker 2'])])
        self.assertEqual(speaker_for(4.1, 4.8, self.turns), 'Speaker 2')

    def test_adjacent_turns_are_not_simultaneous(self):
        turns = [dict(start=0., end=1., speaker='A'), dict(start=1., end=2., speaker='B')]
        self.assertEqual(overlaps(turns), [])

    def test_duplicate_turns_are_not_a_second_person(self):
        self.assertEqual(overlaps([self.turns[0], self.turns[0]]), [])

    def test_three_people_preserve_their_active_intervals(self):
        turns = self.turns + [dict(start=4.5, end=6., speaker='Speaker 3')]
        result = overlaps(turns)
        self.assertEqual([len(x['speakers']) for x in result], [2, 3, 2])

    def test_word_alignment_splits_at_interruption(self):
        result = dict(turns=self.turns, segments=[], words=[
            dict(start=3., end=3.5, text=' Hola'),
            dict(start=4.1, end=4.8, text=' sí'),
            dict(start=6., end=6.5, text=' seguimos')])
        annotate(result)
        self.assertEqual([s['speaker'] for s in result['segments']], ['Speaker 1','Speaker 2','Speaker 1'])
        self.assertEqual(result['segments'][1]['overlapping_speakers'], ['Speaker 1','Speaker 2'])
        self.assertEqual(''.join(s['text'] for s in result['segments']), ' Hola sí seguimos')

    def test_native_chunks_keep_their_text_and_actual_timing(self):
        result = dict(turns=self.turns, words=[], segments=[dict(start=0.,end=10.,text='original')])
        annotate(result)
        self.assertEqual(result['segments'][0]['text'],'original')
        self.assertEqual(result['segments'][0]['end'],10.)
        self.assertEqual(result['segments'][0]['overlapping_speakers'], ['Speaker 1','Speaker 2'])

    def test_profiles_exclude_overlap_and_dont_count_duplicate_turns(self):
        self.assertEqual(clean_seconds('Speaker 1',self.turns,overlaps(self.turns)),9.)
        self.assertEqual(clean_seconds('Speaker 2',self.turns,overlaps(self.turns)),0.)
        duplicate = self.turns + [copy.copy(self.turns[0])]
        self.assertEqual(clean_seconds('Speaker 1',duplicate,overlaps(duplicate)),9.)

    def test_outside_speech_has_no_speaker(self):
        self.assertIsNone(speaker_for(20.,21.,self.turns))

    def test_zero_width_word_keeps_original_segment_text_without_fake_timing(self):
        result=dict(duration=10.,turns=self.turns,segments=[],words=[])
        append_recognition_segment(result,0.,3.,' Fictional complete sentence',[
            dict(start=0.,end=1.,text=' Fictional'),dict(start=1.,end=1.,text=' complete'),
            dict(start=1.,end=3.,text=' sentence')])
        append_recognition_segment(result,6.,7.,' after', [dict(start=6.,end=7.,text=' after')])
        annotate(result)
        self.assertEqual(''.join(s['text'] for s in result['segments']),' Fictional complete sentence after')
        self.assertEqual((result['segments'][0]['start'],result['segments'][0]['end']),(0.,3.))
        self.assertEqual(len(result['words']),3)
        self.assertEqual(result['provenance']['word_alignment']['unaligned_words'],1)
        self.assertFalse(any('_source_segment' in w for w in result['words']))


class ProgressTests(unittest.TestCase):
    def test_community_reports_real_batch_fraction_and_separate_subtasks(self):
        from unittest.mock import patch
        import worker
        with patch.object(worker, 'progress') as report:
            worker.community_progress('embeddings', total=200, completed=50)
            self.assertEqual(report.call_args.args[:2], ('diarizing_progress',25))
            self.assertIn('embeddings', report.call_args.args[2])
            worker.community_progress('embeddings')
            self.assertEqual(report.call_args.args[:2], ('diarizing',0))
            self.assertIn('Grouping speech windows', report.call_args.args[2])
            worker.community_progress('clustering')
            self.assertEqual(report.call_args.args[:2], ('diarizing',0))
            worker.community_progress('segmentation', total=20, completed=20)
            self.assertEqual(report.call_args.args[:2], ('diarizing_progress',100))

    def test_audio_progress_uses_media_time_and_waits_for_success(self):
        from unittest.mock import patch, Mock
        import io
        import worker
        def process(code):
            p=Mock()
            p.stdout=io.StringIO('  Duration: 00:02:00.00, start: 0.0\nout_time_us=30000000\nout_time_us=120000000\n')
            p.wait.return_value=code
            p.poll.return_value=code
            return p
        with patch.object(worker,'ffmpeg_executable',return_value='fixture'), patch.object(worker.subprocess,'Popen',return_value=process(0)), patch.object(worker,'progress') as report:
            worker.convert_audio({},'input','output')
            self.assertEqual([c.args[1] for c in report.call_args_list],[25,99,100])
        with patch.object(worker,'ffmpeg_executable',return_value='fixture'), patch.object(worker.subprocess,'Popen',return_value=process(1)), patch.object(worker,'progress') as report:
            with self.assertRaisesRegex(ValueError,'Audio conversion failed'):
                worker.convert_audio({},'input','output')
            self.assertNotIn(100,[c.args[1] for c in report.call_args_list])


if __name__ == '__main__':
    unittest.main()
