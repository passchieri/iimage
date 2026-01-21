
.PHONY: all init test srcs clean docs publish publish_bin publish_docs lint docker docker_image publish_image install uninstall

PACKAGE_NAME        := $(shell project_info name)
PROJECT_SCRIPTS     := $(shell project_info scripts)
VERSION             := $(shell project_info version)
NORMALIZED_VERSION  := $(shell echo $(VERSION) | sed -e 's/+/_/g')

SRCS           =$(wildcard src/*.py) $(wildcard src/*/*.py) $(wildcard src/*/*/*.py) $(wildcard src/*/*/*/*.py)
INSTALL_SCRIPTS=$(addprefix $(bindir)/,$(PROJECT_SCRIPTS))
VENV_SCRIPTS   =$(addprefix $(datadir)/venv/bin/,$(PROJECT_SCRIPTS))
WHEEL          =$(addprefix dist/, $(PACKAGE_NAME)-$(VERSION)-py3-none-any.whl)
SRCS_TAR       :=$(addprefix dist/,$(PACKAGE_NAME)-$(VERSION).tar.gz)

REPO			= simpartnerregistry.azurecr.io
TAG				= simulytic/ps360metrics

DOCS := dist/doc/usage.md

prefix = /opt
datarootdir = $(prefix)
datadir = $(datarootdir)/$(PACKAGE_NAME)
exec_prefix=/usr/local
bindir=$(exec_prefix)/bin

all: $(WHEEL)

$(WHEEL) $(SRCS_TAR): $(SRCS)
	pytest -v --cov=src --cov-report=xml test
	python -m build
	# Reinstall, because version numbers could have changed
	pip install -e .[dev]

test: srcs
	tox -p

publish: publish_bin publish_docs

publish_bin: $(WHEEL) $(SRCS_TAR)
	python -m twine upload $?

publish_docs: $(DOCS)
	wiki page upload "/Development/Tooling/Python based tooling/${PACKAGE_NAME}" "dist/doc/usage.md" --create_parents

init:
	[ -d src ] || mkdir src
	[ -d test ] || mkdir test
	pip install --upgrade pip
	pip install -e .[dev]

clean:
	find . -name site-packages -prune -o -name __pycache__  -o -name '*.egg-info' -exec rm -Rf '{}' \+
	rm -Rf dist build coverage.xml .coverage .pytest_cache docker/*.whl

docker: publish_image

docker/$(PACKAGE_NAME)-$(VERSION)-py3-none-any.whl: $(WHEEL)
	cp $< $@

docker_image: docker/$(PACKAGE_NAME)-$(VERSION)-py3-none-any.whl
	docker build \
    	--platform=linux/amd64 \
    	-t $(TAG):$(NORMALIZED_VERSION) \
    	-t $(REPO)/$(TAG):$(NORMALIZED_VERSION) \
    	-t $(TAG):latest \
    	-t $(REPO)/$(TAG):latest \
    	--build-arg PIP_EXTRA_INDEX_URL=$PIP_EXTRA_INDEX_URL \
    	-f docker/Dockerfile docker

publish_image: docker_image
	docker push $(REPO)/$(TAG):latest
	docker push $(REPO)/$(TAG):$(NORMALIZED_VERSION)


install: $(INSTALL_SCRIPTS)

srcs: $(SRCS) 

docs: $(DOCS)

$(DOCS): $(WHEEL)
	./utils/create_usage.sh

$(bindir)/%: $(datadir)/venv/bin/%
	ln -fs $< $@
	$@ -h &>/dev/null

$(datadir)/venv:
	mkdir -p $(datadir)
	python3 -m venv $(datadir)/venv
	source $(datadir)/venv/bin/activate && pip install --upgrade pip

$(VENV_SCRIPTS): $(WHEEL) $(datadir)/venv
	source $(datadir)/venv/bin/activate && pip install --force-reinstall $<

uninstall:
	rm -Rf $(datadir)
	rm -f $(INSTALL_SCRIPTS)
