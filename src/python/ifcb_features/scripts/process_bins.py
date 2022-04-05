#!python
"""Load bin, extract features, create blob image, classify samples, and write to disk."""
import logging
import os
from pathlib import Path
from zipfile import ZipFile

import click
import ifcb
import numpy as np
import pandas as pd
from ifcb.data.imageio import format_image
from ifcb_features import classify, compute_features
from PIL import Image


def process_bin(file: Path, outdir: Path, model_config: classify.KerasModelConfig):
    logging.info(f'Processing {file}, saving results to {outdir}')
    if not outdir.exists:
        outdir.mkdir(parents=True)

    bin = ifcb.open_raw(file)

    blobs_fname = outdir / f'{bin.lid}_blobs.zip'
    features_fname = outdir / f'{bin.lid}_features.csv'
    class_fname = outdir / f'{bin.lid}_class_scores.csv'

    features_df = None
    roi_number = None
    num_rois = len(bin.images.keys())
    image_stack = np.zeros((num_rois, model_config.img_dims[0], model_config.img_dims[1], 3), dtype=np.uint8)
    try:
        with ZipFile(blobs_fname, 'w') as blob_zip:
            for ix, roi_number in enumerate(bin.images):
                if ix % 100 == 0:
                    logging.info(f'Processing ROI {roi_number}')

                # Select image
                image = bin.images[roi_number]

                # Compute features
                blob_img, features = compute_features(image)

                # Write blob image to zip as bytes.
                # Include ROI number in filename. e.g. D20141117T234033_IFCB102_2.png
                image_bytes = blob2bytes(blob_img)
                blob_zip.writestr(f'{bin.pid.with_target(roi_number)}.png', image_bytes)

                # Add features row to dataframe
                # - Copied pyifcb
                row_df = features2df(features, roi_number)
                if features_df is None:
                    features_df = row_df
                else:
                    features_df = pd.concat([features_df, row_df])

                # Resize image, normalized, and add to stack
                pil_img = (Image
                    .fromarray(image)
                    .convert('RGB')
                    .resize(model_config.img_dims, Image.BILINEAR)
                )
                img = np.array(pil_img) / model_config.norm
                image_stack[ix, :] = img
    except Exception as e:
        logging.error(f'Failed to process {file} for ROI {roi_number}')
        if os.path.exists(blobs_fname):
            os.remove(blobs_fname)
        raise e

    # Save features dataframe
    if features_df is not None:
        logging.info(f'Saving features to {features_fname}')
        features_df.to_csv(features_fname, index=False, float_format='%.6f')
        # Classify images and save as csv
        logging.info(f'Classifying images and saving to {class_fname}')
        _ = classify.predict(model_config, image_stack, class_fname)
    else:
        logging.info(f'No features found in {file}. Skipping classification.')



def blob2bytes(blob_img: np.ndarray) -> bytes:
    """Format blob as image to be written in zip file."""
    image_buf = format_image(blob_img)
    return image_buf.getvalue()


def features2df(features: list, roi_number: int) -> pd.DataFrame:
    """Convert features to dataframe (copy pasta from featureio.py)."""
    cols, values = zip(*features)
    cols = ('roiNumber',) + cols
    values = (roi_number,) + values
    values = [(value,) for value in values]
    return pd.DataFrame({c: v for c, v in zip(cols, values)},
                        columns=cols)


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('class_path', type=click.Path(exists=True))
def cli(input_dir: Path, output_dir: Path, model_path: Path, class_path: Path):
    """Process all files in input_dir and write results to output_dir."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    model_config = classify.KerasModelConfig(model_path=model_path, class_path=class_path)
    for file in input_dir.glob('*.adc'):
        process_bin(file, output_dir, model_config)


#if __name__ == '__main__':
#    cli()
