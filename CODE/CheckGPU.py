import tensorflow as tf

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Num GPUs Available: ", len(gpus))
        for gpu in gpus:
            print("GPU Name: ", gpu.name)
    else:
        print("No GPU available. Using CPU.")
        print("Num GPUs Available: 0")


if __name__ == "__main__":
    check_gpu()

