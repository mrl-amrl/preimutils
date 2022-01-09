rm -rf build dist preimutils.egg-info
python3 setup.py sdist bdist_wheel
pip install --upgrade .
