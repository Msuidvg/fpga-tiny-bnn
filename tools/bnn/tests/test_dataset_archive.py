from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools.bnn.train_bnn import load_dataset

class ArchiveTests(unittest.TestCase):
    def test_rejects_nonbinary_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"invalid.npz"
            np.savez(p,images=np.full((1,32,32),2),labels=[1],batches=["a"],paths=["a/1.png"])
            with self.assertRaisesRegex(ValueError,"binary"):
                load_dataset(p)

    def test_keeps_labels_batches_and_pixel_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"valid.npz"
            masks=np.zeros((2,32,32),dtype=np.uint8);masks[1,3,7]=1
            np.savez(p,images=masks,labels=[1,5],batches=["a","b"],paths=["a/1.png","b/5.png"])
            data=load_dataset(p)
            np.testing.assert_array_equal(data.images,masks)
            self.assertEqual(data.labels.tolist(),[1,5])
            self.assertEqual(data.batches.tolist(),["a","b"])
