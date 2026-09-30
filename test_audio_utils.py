import unittest
import numpy as np
from audio_utils import prepare_audio, append_text


class AudioTests(unittest.TestCase):
    def test_stereo_pcm_normalization(self):
        audio, rate = prepare_audio((16000, np.array([[-32768, 32767], [16384, 16384]], dtype=np.int16)))
        self.assertEqual(rate, 16000)
        np.testing.assert_allclose(audio, [-1 / 65536, 0.5])

    def test_reject_bad_audio(self):
        for audio in [None, (16000, np.array([])), (0, np.zeros(5)), (16000, np.array([np.nan])), (1, np.zeros(61))]:
            with self.subTest(audio=audio), self.assertRaises(ValueError):
                prepare_audio(audio)

    def test_preserve_previous_text(self):
        self.assertEqual(append_text("Previous text", " new words "), "Previous text\nnew words")
        self.assertEqual(append_text("Previous text", " "), "Previous text")


if __name__ == "__main__":
    unittest.main()
