clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

u:
	pip uninstall torch_pointcloud

i:
	# python setup.py install
	pip install -v -e .
