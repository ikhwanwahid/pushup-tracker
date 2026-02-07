"""Text feedback generation from push-up quality scores."""

from src.quality.scoring import RepScore


def generate_feedback(score: RepScore) -> list[str]:
    """Generate actionable text feedback from a rep score.

    Args:
        score: RepScore from QualityScorer.

    Returns:
        List of feedback strings, most important first.
    """
    feedback = []

    # Back alignment feedback
    if score.back_alignment < 50:
        feedback.append(
            "Keep your back straight — avoid sagging your hips or piking up. "
            "Engage your core throughout the movement."
        )
    elif score.back_alignment < 75:
        feedback.append(
            "Your back alignment could improve. Focus on maintaining a straight "
            "line from shoulders to ankles."
        )

    # Depth feedback
    if score.depth < 50:
        feedback.append(
            "Go deeper — your elbows should bend to about 90 degrees at the "
            "bottom of each rep."
        )
    elif score.depth < 75:
        feedback.append(
            "Try to go a bit deeper on each rep for full range of motion."
        )

    # Extension feedback
    if score.extension < 50:
        feedback.append(
            "Fully extend your arms at the top of each rep — lock out without "
            "hyperextending."
        )
    elif score.extension < 75:
        feedback.append(
            "Almost full extension at the top — push up a little more."
        )

    # Overall encouragement
    if score.composite >= 85:
        feedback.insert(0, "Excellent form! Keep it up.")
    elif score.composite >= 70:
        feedback.insert(0, "Good form overall with some room for improvement.")
    elif score.composite >= 50:
        feedback.insert(0, "Decent effort — focus on the areas below to improve.")
    else:
        feedback.insert(0, "Several aspects of your form need attention.")

    if not feedback:
        feedback.append("Great push-up form!")

    return feedback
