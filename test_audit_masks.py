import unittest
import numpy as np
from extend_slake_audit import label_plane


class MaskDecoding(unittest.TestCase):
    def test_opaque_repeated_rgba_preserves_label_values(self):
        a=np.array([[0,70],[95,185]],dtype=np.uint8)
        rgba=np.stack([a,a,a,np.full_like(a,255)],axis=-1)
        self.assertTrue(np.array_equal(label_plane(rgba),a))

    def test_color_or_transparent_masks_not_guessed(self):
        color=np.array([[[1,2,3,255]]],dtype=np.uint8)
        transparent=np.array([[[70,70,70,1]]],dtype=np.uint8)
        self.assertIsNone(label_plane(color))
        self.assertIsNone(label_plane(transparent))
