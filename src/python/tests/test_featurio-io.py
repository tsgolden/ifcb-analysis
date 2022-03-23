# Test script to extract blobs, features, classify samples, and write to disk.
from pathlib import Path

import ifcb
import numpy as np
import pandas as pd
from ifcb.data.imageio import format_image
from ifcb_features import compute_features


class TestFeatures:

    basedir = Path(__file__).parent / 'data'
    adc_file = basedir / 'D20141117T234033_IFCB102.adc'
    hdf_file = basedir / 'D20141117T234033_IFCB102.hdf'
    roi_file = basedir / 'D20141117T234033_IFCB102.roi'

    def _pack_df(self, features, roi):
        cols, values = zip(*features)
        cols = ('roiNumber',) + cols
        values = (roi,) + values
        values = [(value,) for value in values]
        return pd.DataFrame(
            {c: v for c, v in zip(cols, values)},
            columns=cols
        )

    def test_process_bin(self):
        # Give ADC file
        bin = ifcb.open_raw(self.adc_file)
        PID = 'D20141117T234033_IFCB102'
        N_ROIS = 1346
        N_FEATURES = 240
        IMG_SHAPE = (72, 80)
        ROI = 2

        # Check pid/lid
        assert str(bin.lid) == PID
        assert str(bin.pid) == PID

        # Check number of samples is correct
        assert len(bin.images.keys()) == N_ROIS

        # Check that image is correct
        assert bin.images[2].shape == IMG_SHAPE
        assert bin.images[2].dtype == np.uint8

        # Check features and blob
        blob_img, features = compute_features(bin.images[ROI])
        assert np.sum(blob_img) == 389
        assert blob_img.dtype == np.bool8
        assert blob_img.shape == IMG_SHAPE

        df = self._pack_df(features, ROI)
        assert len(features) == N_FEATURES
        # Adds ROI number to features, so n + 1 features
        assert df.shape == (1, N_FEATURES+1)
