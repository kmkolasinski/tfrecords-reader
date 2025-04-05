from simple_app.main import main


def test__main__check_output():
    """Test the main function."""
    assert main() == "Hello, this is the empty service."
