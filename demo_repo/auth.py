USERS = {
    "admin": {
        "password": "1234",
        "failed_attempts": 0,
    }
}


def login(username: str, password: str):
    """
    Authenticate a user using username and password.
    """

    user = USERS.get(username)

    if not user:
        return {
            "success": False,
            "error": "User not found",
        }

    if user["password"] != password:
        user["failed_attempts"] += 1

        return {
            "success": False,
            "error": "Invalid credentials",
        }

    user["failed_attempts"] = 0

    token = generate_token(username)

    return {
        "success": True,
        "token": token,
    }


def generate_token(username: str):
    """
    Create a simple authentication token.
    """

    return f"token-{username}-123"


def validate_token(token: str):
    """
    Check whether the supplied token has
    the expected format.
    """

    if not token:
        return False

    return token.startswith("token-")


def logout(token: str):
    """
    Log the user out.
    """

    if validate_token(token):
        return {
            "success": True,
            "message": "Logged out",
        }

    return {
        "success": False,
        "error": "Invalid token",
    }


def reset_password(username: str, new_password: str):
    """
    Change the password for an existing user.
    """

    user = USERS.get(username)

    if not user:
        return {
            "success": False,
            "error": "User not found",
        }

    user["password"] = new_password

    return {
        "success": True,
        "message": "Password updated",
    }