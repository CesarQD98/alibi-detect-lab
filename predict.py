from alibi_detect.utils.saving import (
    save_detector,
    load_detector,
)
from matplotlib import pyplot as plt  # Use this if the above line does not work.
from alibi_detect.od import OutlierAE, OutlierVAE
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image

# initialize outlier detector
od = OutlierVAE(
    threshold=0.015,  # threshold for outlier score above which the element is flagged as an outlier.
    score_type="mse",  # use MSE of reconstruction error for outlier detection
    encoder_net=encoder_net,  # can also pass VAE model instead
    decoder_net=decoder_net,  # of separate encoder and decoder
    latent_dim=latent_dim,
    samples=4,
)

save_detector(od, "saved_outlier_models/carpet_od_20epochs.h5")
# od = load_detector(filepath)

# Test our model on a bad image
img_num = 9
test_bad_image = bad_dataset[img_num].reshape(1, 64, 64, 3)
plt.imshow(test_bad_image[0])

test_bad_image_recon = od.vae(test_bad_image)
test_bad_image_recon = test_bad_image_recon.numpy()
plt.imshow(test_bad_image_recon[0])

test_bad_image_predict = od.predict(
    test_bad_image
)  # Returns a dictionary of data and metadata

# Data dictionary contains the instance_score, feature_score, and whether it is an outlier or not.
# Let u look at the values under the 'data' key in our output dictionary
bad_image_instance_score = test_bad_image_predict["data"]["instance_score"][0]
print("The instance score is:", bad_image_instance_score)

bad_image_feature_score = test_bad_image_predict["data"]["feature_score"][0]
plt.imshow(bad_image_feature_score[:, :, 0])
print(
    "Is this image an outlier (0 for NO and 1 for YES)?",
    test_bad_image_predict["data"]["is_outlier"][0],
)

# You can also manually define the threshold based on your specific use case.
od.threshold = 0.002
print("Current threshld value is: ", od.threshold)

# Let us check it for multiple images
X = bad_dataset[:20]

od_preds = od.predict(
    X,
    outlier_type="instance",  # use 'feature' or 'instance' level
    return_feature_score=True,  # scores used to determine outliers
    return_instance_score=True,
)

print(list(od_preds["data"].keys()))

# Scatter plot of instance scores. using the built-in function for the scatterplot.
target = np.ones(
    X.shape[0],
).astype(
    int
)  # Ground truth (all ones for bad images)
labels = ["normal", "outlier"]
plot_instance_score(
    od_preds, target, labels, od.threshold
)  # pred, target, labels, threshold

# Plot features for select images, using the built in function (plot_feature_outlier_image)
X_recon = od.vae(X).numpy()
plot_feature_outlier_image(
    od_preds,
    X,
    X_recon=X_recon,
    instance_ids=[0, 5, 10, 15, 17],  # pass a list with indices of instances to display
    max_instances=5,  # max nb of instances to display
    outliers_only=False,
)  # only show outlier predictions
