from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


def get_files(folder: str) -> list:
    """
    git files from directory
    :return:
    """
    return [f for f in listdir(folder) if isfile(join(folder, f))]


def run():
    data_folder = "data"
    dest_folder = "data_prepared"
    THRESHOLD_VALUE = 254

    for x, f in enumerate(get_files(folder=data_folder)):
        # img = "data/15033728.png"

        img = join(data_folder, f)

        with Image.open(img) as im:
            plt.axis("off")
            im.convert("L")
            imgData = np.asarray(im)
            thresholdedData = (imgData > THRESHOLD_VALUE) * 1.0

            plt.imshow(X=thresholdedData)
            plt.savefig(join(dest_folder, f), transparent=True, bbox_inches="tight", pad_inches=0)
            plt.clf()


if __name__ == "__main__":
    run()
