ASSISTANT_TARGET_MAP = {
    # Negative affect → support
    "guilty": "caring",
    "ashamed": "caring",
    "sad": "caring",
    "lonely": "caring",
    "disappointed": "caring",
    "hopeless": "caring",
    "heartbroken": "caring",
    "mourning": "caring",
    "afraid": "caring",
    "terrified": "caring",
    "apprehensive": "caring",
    "angry": "caring",
    "furious": "caring",
    "frustrated": "caring",
    "annoyed": "caring",
    "jealous": "caring",
    "disgusted": "caring",
    "embarrassed": "caring",
    # Positive affect → amplify positivity
    "confident": "admiration",
    "excited": "joy",
    "happy": "joy",
    "joyful": "joy",
    "proud": "admiration",
    "grateful": "gratitude",
    "thrilled": "joy",
    "ecstatic": "joy",
    # Surprise → shared positive surprise
    "surprised": "joy",
    "astonished": "joy",
}


def get_assistant_target(user_emotion: str) -> str:
    """
    Maps user emotion → assistant emotional target.
    Defaults to 'caring' if unknown.
    """
    return ASSISTANT_TARGET_MAP.get(user_emotion, "caring")
