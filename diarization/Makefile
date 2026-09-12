build:
	@echo "Building..."
	@go build -o main cmd/cli/.

run:
	@go run -race ./cmd/playground/.

test:
	@echo "Testing..."
	@go test ./... -v

# Clean the binary
clean:
	@echo "Cleaning..."
	@rm -f main

# Remove unused dep
tidy:
	@go mod tidy
