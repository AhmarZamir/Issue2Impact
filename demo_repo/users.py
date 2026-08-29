USERS = {
    1: {
        "name": "John",
        "email": "john@example.com"
    }
}


def get_user(user_id: int):
    return USERS.get(user_id)