import pytest

def test_requirements():
    with open('requirements-dev.txt') as f:
        requirements = f.readlines()
    assert 'pytest' in requirements
    assert 'requests' in requirements  # Update this line based on actual dependencies
    assert 'flask' in requirements  # Update this line based on actual dependencies