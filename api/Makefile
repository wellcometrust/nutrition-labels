AWS_ACCOUNT_ID := 160358319781
IMAGE := org.wellcome/ml-services
TAG := nutrition-labels
VERSION := 2021.7.0
ECR_IMAGE := $(AWS_ACCOUNT_ID).dkr.ecr.eu-west-1.amazonaws.com/$(IMAGE)

.PHONY: docker-build
docker-build:
	docker build -t $(ECR_IMAGE):$(TAG)-$(VERSION) \
                     -t $(ECR_IMAGE):$(TAG) \
                     -f Dockerfile .
.PHONY: aws-docker-login
aws-docker-login:
	aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.eu-west-1.amazonaws.com

.PHONY: docker-push
docker-push: docker-build
        aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.eu-west-1.amazonaws.com \
        && docker push $(ECR_IMAGE):$(TAG)-$(VERSION) \ 
        && docker push $(ECR_IMAGE):$(TAG)                                      
