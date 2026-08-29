from auth import login
from users import get_user


def authenticate_user(username, password):
    return login(username, password)


def user_profile(user_id):
    return get_user(user_id)