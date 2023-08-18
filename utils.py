from neural_nets.emotions import predict_emotions
from neural_nets.inappropriate import predict_appropriateness
from neural_nets.sentiment import predict_sentiment
from neural_nets.toxicity import predict_toxicity
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def collect_profile(phrases):
    profile_stats = [
        "Нейтральность",
        "Радость",
        "Грусть",
        "Удивление",
        "Страх",
        "Гнев",
        "Неуместные высказываения",
        "Негативный настрой",
        "Токсичные сообщения",
    ]
    user_profile = {key: 0 for key in profile_stats}
    for phrase in phrases:
        message_profile = predict_emotions(phrase)
        message_profile["Неуместные высказываения"] = predict_appropriateness(phrase)[
            "Inappropriate"
        ]
        message_profile["Негативный настрой"] = predict_sentiment(phrase)["NEGATIVE"]
        message_profile["Токсичные сообщения"] = predict_toxicity(phrase)
        for k, v in message_profile.items():
            user_profile[k] += v
    return {k: v / len(phrases) for k, v in user_profile.items()}


def create_profile_image(profile, save_path):
    labels = list(profile.keys())
    values = [val * 100 for val in profile.values()]

    # Number of variables we're plotting.
    num_vars = len(labels)

    # Split the circle into even parts and save the angles
    # so we know where to put each axis.
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    # and append the start value to the end.
    values += values[:1]
    angles += angles[:1]

    # ax = plt.subplot(polar=True)
    fig, ax = plt.subplots(figsize=(9, 6), subplot_kw=dict(polar=True))
    # Draw the outline of our data.
    ax.plot(angles, values, color="red", linewidth=1)
    # Fill it in.
    ax.fill(angles, values, color="red", alpha=0.25)

    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)

    # Go through labels and adjust alignment based on where
    # it is in the circle.
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment("center")
        elif 0 < angle < np.pi:
            label.set_horizontalalignment("left")
        else:
            label.set_horizontalalignment("right")
    # Ensure radar goes from 0 to 100.
    ax.set_ylim(0, 100)
    # You can also set gridlines manually like this:
    # ax.set_rgrids([20, 40, 60, 80, 100])

    # Set position of y-labels (0-100) to be in the middle
    # of the first two axes.
    ax.set_rlabel_position(180 / num_vars)
    plt.savefig(save_path)
    plt.close()



