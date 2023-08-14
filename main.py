import os
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

print("tensorflow version → ", tf.__version__)

from keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dense,
    Reshape,
    InputLayer,
    Flatten,
)

from alibi_detect.od import OutlierAE, OutlierVAE
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image

from alibi_detect.utils.saving import (
    save_detector,
    load_detector,
)  # Use this if the above line does not work.

##########################################################################
# Load data. We only need good data and anything NOT good is an outlier.

image_directory = "media/antamina/train/"
SIZE = 64
# dataset = (
#     []
# )  # Many ways to handle data, you can use pandas. Here, we are using a list format.


def image_loader():
    data_dir = Path("media/antamina/train")

    dataset = []
    bad_dataset = []
    threshold_infer_dataset = []

    good_images_folder = Path(data_dir, "good")
    bad_images_folder = Path(data_dir, "bad")
    threshold_infer_images_folder = Path(data_dir, "threshold")

    good_files = list(good_images_folder.glob("*"))
    bad_files = list(bad_images_folder.glob("*"))
    threshold_files = list(threshold_infer_images_folder.glob("*"))

    for file in good_files:
        image = cv2.imread(str(file))
        image = Image.fromarray(image, "RGB")
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))

    dataset = np.array(dataset)

    for file in bad_files:
        image = cv2.imread(str(file))
        image = Image.fromarray(image, "RGB")
        image = image.resize((SIZE, SIZE))
        bad_dataset.append(np.array(image))

    bad_dataset = np.array(bad_dataset)

    for file in threshold_files:
        image = cv2.imread(str(file))
        image = Image.fromarray(image, "RGB")
        image = image.resize((SIZE, SIZE))
        threshold_infer_dataset.append(np.array(image))

    threshold_infer_dataset = np.array(threshold_infer_dataset)

    print("len(threshold_infer_dataset) → ", len(threshold_infer_dataset))

    dataset = dataset.astype("float32") / 255.0
    bad_dataset = bad_dataset.astype("float32") / 255.0
    threshold_infer_dataset = threshold_infer_dataset.astype("float32") / 255.0

    return dataset, bad_dataset, threshold_infer_dataset

    # good_images = []

    # dataset = []
    # good_images = os.listdir(image_directory + "good/")
    # for i, image_name in enumerate(good_images):
    #     if image_name.split(".")[1] == "jpg" or image_name.split(".")[1] == "jpeg":
    #         image = cv2.imread(image_directory + "good/" + image_name)
    #         image = Image.fromarray(image, "RGB")
    #         image = image.resize((SIZE, SIZE))
    #         dataset.append(np.array(image))

    # dataset = np.array(dataset)
    # print("dataset antes → ", len(dataset))

    # train = dataset[0:300]
    # test = dataset[300:]
    # print("train antes → ", len(train))
    # print("test antes → ", len(test))

    # # Let us also load bad images to verify our trained model.
    # bad_images = os.listdir(image_directory + "bad")
    # bad_dataset = []
    # for i, image_name in enumerate(bad_images):
    #     if image_name.split(".")[1] == "jpg" or image_name.split(".")[1] == "jpeg":
    #         image = cv2.imread(image_directory + "bad/" + image_name)
    #         image = Image.fromarray(image, "RGB")
    #         image = image.resize((SIZE, SIZE))
    #         bad_dataset.append(np.array(image))
    # print("len(bad_dataset) → ", len(bad_dataset))
    # bad_dataset = np.array(bad_dataset)
    # print("bad_dataset.size → ", bad_dataset.size)

    # gaaa = np.append(test, bad_dataset)
    # print("gaaa despues → ", gaaa.size)

    # train = train.astype("float32") / 255.0
    # test = test.astype("float32") / 255.0
    # bad_dataset = bad_dataset.astype("float32") / 255.0

    # return train, test, bad_dataset


#########################################################################
# Define the encoder - decoder network for input to the OutlierVAE detector class.
# Can be any encoder and decoder.


def encoder_decoder_generator(dataset):
    encoding_dim = 1024  # Dimension of the bottleneck encoder vector.
    dense_dim = [
        8,
        8,
        512,
    ]  # Dimension of the last conv. output. This is used to work our way back in the decoder.

    # Define encoder
    encoder_net = tf.keras.Sequential(
        [
            InputLayer(input_shape=dataset[0].shape),
            Conv2D(64, 4, strides=2, padding="same", activation=tf.nn.relu),
            Conv2D(128, 4, strides=2, padding="same", activation=tf.nn.relu),
            Conv2D(512, 4, strides=2, padding="same", activation=tf.nn.relu),
            Flatten(),
            Dense(
                encoding_dim,
            ),
        ]
    )

    print(encoder_net.summary())
    # print(encoder_net.input_shape)

    # Define the decoder.
    # Start with the bottleneck dimension (encoder vector) and connect to dense layer
    # with dim = total nodes in the last conv. in the encoder.
    decoder_net = tf.keras.Sequential(
        [
            InputLayer(input_shape=(encoding_dim,)),
            Dense(np.prod(dense_dim)),
            Reshape(target_shape=dense_dim),
            Conv2DTranspose(256, 4, strides=2, padding="same", activation=tf.nn.relu),
            Conv2DTranspose(64, 4, strides=2, padding="same", activation=tf.nn.relu),
            Conv2DTranspose(3, 4, strides=2, padding="same", activation="sigmoid"),
        ]
    )

    print(decoder_net.summary())
    # print(decoder_net.input_shape)

    return encoder_net, decoder_net


def outlier_train(encoder_net, decoder_net, dataset):
    #######################################################################
    # Define and train the outlier detector.

    latent_dim = 1024  # (Same as encoding dim. )

    # initialize outlier detector
    od = OutlierVAE(
        threshold=0.015,  # threshold for outlier score above which the element is flagged as an outlier.
        score_type="mse",  # use MSE of reconstruction error for outlier detection
        encoder_net=encoder_net,  # can also pass VAE model instead
        decoder_net=decoder_net,  # of separate encoder and decoder
        latent_dim=latent_dim,
        samples=4,
    )

    print("Current threshold value is: ", od.threshold)

    # train
    # from alibi_detect.models.tensorflow.losses import elbo #evidence lower bound loss

    adam = tf.keras.optimizers.Adam(lr=1e-4)

    od.fit(dataset, optimizer=adam, epochs=20, batch_size=4, verbose=True)

    # Check the threshold value. Should be the same as defined before.
    print("After training, current threshold value is: ", od.threshold)

    #
    # infer_threshold Updates threshold by a value inferred from the percentage of
    # instances considered to be outliers in a sample of the dataset.
    # percentage of X considered to be normal based on the outlier score.
    # Here, we set it to 99%
    # od.infer_threshold(test, outlier_type="instance", threshold_perc=99.0)
    # print("Current threshold value is: ", od.threshold)

    # ######################################################################

    # save the trained outlier detector

    # TODO: Unique filename creator

    save_detector(od, "saved_outlier_models/antamina_1.h5")
    # od = load_detector("./saved_outlier_models/carpet_od_20epochs.h5")

    return od


def predict(image_dataset, od):
    for image in image_dataset:
        image_dataset = image.reshape(1, 64, 64, 3)
    print(len(image_dataset))

    prediction = od.predict(image_dataset)

    prediction_scores = prediction["data"]["instance_score"]
    print("len(prediction) → ", len(prediction))
    print(prediction_scores)

    print()
    print(len(prediction["data"]["feature_score"]))


def main():
    dataset, bad_dataset, threshold_infer_dataset = image_loader()
    encoder_net, decoder_net = encoder_decoder_generator(dataset=dataset)
    # od = outlier_train(
    #     encoder_net=encoder_net, decoder_net=decoder_net, dataset=dataset
    # )

    od = load_detector("./saved_outlier_models/antamina_1.h5")
    print()
    print("********************")
    print("Current threshold value is (before infer) → ", od.threshold)

    od.infer_threshold(
        threshold_infer_dataset, outlier_type="instance", threshold_perc=71.4
    )
    print()
    print("********************")
    print("Current threshold value is (after infer) → ", od.threshold)

    predict(image_dataset=threshold_infer_dataset, od=od)


if __name__ == "__main__":
    main()

# od.threshold = 0.019

# # Test our model on a bad image
# img_num = 9
# test_bad_image = bad_dataset[img_num].reshape(1, 64, 64, 3)
# plt.imshow(test_bad_image[0])

# test_bad_image_recon = od.vae(test_bad_image)
# test_bad_image_recon = test_bad_image_recon.numpy()
# plt.imshow(test_bad_image_recon[0])

# test_bad_image_predict = od.predict(
#     test_bad_image
# )  # Returns a dictionary of data and metadata

# # Data dictionary contains the instance_score, feature_score, and whether it is an outlier or not.
# # Let u look at the values under the 'data' key in our output dictionary
# bad_image_instance_score = test_bad_image_predict["data"]["instance_score"][0]
# print("The instance score is:", bad_image_instance_score)

# bad_image_feature_score = test_bad_image_predict["data"]["feature_score"][0]
# plt.imshow(bad_image_feature_score[:, :, 0])
# print(
#     "Is this image an outlier (0 for NO and 1 for YES)?",
#     test_bad_image_predict["data"]["is_outlier"][0],
# )

# # You can also manually define the threshold based on your specific use case.
# od.threshold = 0.002
# print("Current threshold value is: ", od.threshold)

# # Let us check it for multiple images
# X = bad_dataset[:20]

# od_preds = od.predict(
#     X,
#     outlier_type="instance",  # use 'feature' or 'instance' level
#     return_feature_score=True,  # scores used to determine outliers
#     return_instance_score=True,
# )

# print(list(od_preds["data"].keys()))

# # Scatter plot of instance scores. using the built-in function for the scatterplot.
# target = np.ones(
#     X.shape[0],
# ).astype(
#     int
# )  # Ground truth (all ones for bad images)
# labels = ["normal", "outlier"]
# plot_instance_score(
#     od_preds, target, labels, od.threshold
# )  # pred, target, labels, threshold

# # Plot features for select images, using the built in function (plot_feature_outlier_image)
# X_recon = od.vae(X).numpy()
# plot_feature_outlier_image(
#     od_preds,
#     X,
#     X_recon=X_recon,
#     instance_ids=[0, 5, 10, 15, 17],  # pass a list with indices of instances to display
#     max_instances=5,  # max nb of instances to display
#     outliers_only=False,
# )  # only show outlier predictions

# #######################################
