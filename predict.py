from main import bad_dataset, plt, load_detector

od = load_detector("./saved_outlier_models/antamina_1.h5")

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

bad_image_instance_score = test_bad_image_predict["data"]["instance_score"][0]
print("The instance score is:", bad_image_instance_score)

bad_image_feature_score = test_bad_image_predict["data"]["feature_score"][0]
plt.imshow(bad_image_feature_score[:, :, 0])
print(
    "Is this image an outlier (0 for NO and 1 for YES)?",
    test_bad_image_predict["data"]["is_outlier"][0],
)
