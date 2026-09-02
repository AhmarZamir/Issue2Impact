from demo_repo.auth import login


def test_valid_login():
    result = login("admin", "1234")
    assert result["success"] is True


def test_invalid_login():
    result = login("admin", "wrong")
    assert result["success"] is False
