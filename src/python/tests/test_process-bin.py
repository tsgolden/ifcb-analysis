import filecmp
import os
from pathlib import Path

import ifcb
import numpy as np
import pandas as pd
import tensorflow as tf
from click.testing import CliRunner
from ifcb_features import classify, compute_features
from ifcb_features.scripts.process_bins import cli
from PIL import Image


class TestFeatures:

    basedir = Path(__file__).parent / 'data'
    adc_file = basedir / 'D20141117T234033_IFCB102.adc'
    hdf_file = basedir / 'D20141117T234033_IFCB102.hdf'
    roi_file = basedir / 'D20141117T234033_IFCB102.roi'
    model_path = basedir / 'scwharf-ifcb-xception'
    classes_path = basedir / 'scwharf-class-names.txt'
    reference_blobs = basedir / 'reference' / 'D20141117T234033_IFCB102_blobs_v2.zip'
    reference_classes = basedir / 'reference' / 'D20141117T234033_IFCB102_class.h5'
    reference_features = basedir / 'reference' / 'D20141117T234033_IFCB102_fea_v2.csv'

    def _pack_df(self, features, roi):
        cols, values = zip(*features)
        cols = ('roiNumber',) + cols
        values = (roi,) + values
        values = [(value,) for value in values]
        return pd.DataFrame(
            {c: v for c, v in zip(cols, values)},
            columns=cols
        )

    def test_process_image(self):
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

    def test_classify(self):
        bin = ifcb.open_raw(self.adc_file)
        ROI = 2

        model_config = classify.KerasModelConfig(self.model_path, self.classes_path, 'test')
        img = (Image
            .fromarray(bin.images[ROI])
            .convert('RGB')
            .resize(model_config.img_dims, Image.BILINEAR)
        )
        # expecting (1, 299, 299, 3)
        input_array = tf.keras.preprocessing.image.img_to_array(img)
        # predict will not normalize the image, this test model used 255.
        img = input_array[np.newaxis, :] / 255

        predictions_df = classify.predict(model_config, img)
        assert predictions_df.iloc[0].argmax() == 29

    def test_script(self):
        runner = CliRunner()
        result = runner.invoke(cli, [str(self.basedir), str(self.basedir), str(self.model_path), str(self.classes_path), 'test'])
        features_file = self.basedir / 'D20141117T234033_IFCB102_fea_v2.csv'
        classes_file = self.basedir / 'D20141117T234033_IFCB102_class.h5'
        blob_file = self.basedir / 'D20141117T234033_IFCB102_blobs_v2.zip'

        assert result.exit_code == 0
        assert filecmp.cmp(self.reference_features, features_file)
        assert filecmp.cmp(self.reference_classes, classes_file)
        assert self.reference_blobs.stat().st_size == blob_file.stat().st_size

        if result.exit_code == 0:
            os.remove(str(self.basedir / 'D20141117T234033_IFCB102_fea_v2.csv'))
            os.remove(str(self.basedir / 'D20141117T234033_IFCB102_class.h5'))
            os.remove(str(self.basedir / 'D20141117T234033_IFCB102_blobs_v2.zip'))
