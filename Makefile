SOURCES := $(shell find Sources -name '*.swift')
FLAGS := -parse-as-library -target arm64-apple-macos14.0 -swift-version 5
CORE := Sources/FaceUnlock/Core/FaceAligner.swift Sources/FaceUnlock/Core/FaceGeometry.swift Sources/FaceUnlock/Core/VectorMath.swift
CODESIGN_ID ?= -

.PHONY: all release model test model-test enrollment-test smoke run install clean
all: release
release:
	CODESIGN_ID="$(CODESIGN_ID)" python3 scripts/build_app.py
model:
	./scripts/setup_model.sh
test: enrollment-test
	@mkdir -p build
	swiftc $(FLAGS) Sources/JoFaceGuard/GuardPolicy.swift Tests/PolicyTests.swift -o build/policy-tests
	./build/policy-tests
	swiftc $(FLAGS) $(CORE) Sources/JoFaceGuard/FacePipeline.swift Tests/QualityTests.swift Tests/AlignmentTests.swift Tests/VisionTestMain.swift -o build/vision-tests
	./build/vision-tests
model-test:
	@mkdir -p build
	swiftc $(FLAGS) -framework CoreML Sources/JoFaceGuard/EmbeddingModel.swift Tests/ModelTests.swift -o build/model-tests
	JO_FACE_GUARD_MODEL="$(CURDIR)/Resources/Models/SFace.mlpackage" ./build/model-tests Tests/Fixtures/sface-reference.json
enrollment-test:
	@mkdir -p build
	swiftc $(FLAGS) -framework CoreML $(CORE) Sources/JoFaceGuard/FacePipeline.swift Sources/JoFaceGuard/EmbeddingModel.swift Tests/EnrollmentRegressionTests.swift -o build/enrollment-regression-tests
	./build/enrollment-regression-tests Tests/Fixtures/astronaut.png
smoke:
	python3 scripts/build_app.py --smoke-only
install:
	@test ! -e /Applications/JoFaceGuard.app || (echo 'JoFaceGuard already exists in Applications; quit it and move the old app aside first.'; exit 1)
	ditto -xk build/JoFaceGuard.zip /Applications
	codesign --verify --deep --strict /Applications/JoFaceGuard.app
run:
	open /Applications/JoFaceGuard.app
clean:
	rm -rf build
