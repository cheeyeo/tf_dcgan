import glob

import imageio


if __name__ == "__main__":
    anime_file = "dcgan.gif"

    with imageio.get_writer(anime_file, mode="I") as writer:
        filenames = glob.glob("output/plots/*.png")
        filenames = sorted(filenames)

        for f in filenames:
            image = imageio.imread(f)
            writer.append_data(image)
