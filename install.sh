rm -rf build dist preimutils.egg-info
python setup.py sdist bdist_wheel
pip install --upgrade .
