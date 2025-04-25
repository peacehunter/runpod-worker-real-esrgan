group "default" {
  targets = ["app", "base"]
}

target "base" {
  context = "."
  dockerfile = "Dockerfile"
  tags = ["tuhin458/base:latest"]
}

target "app" {
  context = "."
  dockerfile = "Dockerfile"
  tags = ["tuhin458/app:latest"]
  args = {
    VERSION = "1.0"
  }
}
