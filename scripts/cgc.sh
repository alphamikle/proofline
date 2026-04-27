#!/usr/bin/env bash
set -Eeuo pipefail

CGC_HOME="${CGC_HOME:-$HOME/.codegraphcontext}"
CGC_VENV="${CGC_VENV:-$CGC_HOME/cgc-venv}"
CGC_BIN_DIR="${CGC_BIN_DIR:-$HOME/.local/bin}"
CGC_BIN="$CGC_BIN_DIR/cgc"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
CGC_PYTHON_VERSION="${CGC_PYTHON_VERSION:-3.12}"

NEO4J_CONTAINER_NAME="${NEO4J_CONTAINER_NAME:-cgc-neo4j}"
NEO4J_IMAGE="${NEO4J_IMAGE:-neo4j:5-community}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-codegraphcontext}"
NEO4J_DATABASE="${NEO4J_DATABASE:-neo4j}"
NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_HTTP_PORT="${NEO4J_HTTP_PORT:-7474}"
NEO4J_BOLT_PORT="${NEO4J_BOLT_PORT:-7687}"
PYTHON_CMD="${PYTHON_CMD:-}"
SCIP_LANGUAGES="${SCIP_LANGUAGES:-python,typescript,javascript,go,rust,java,scala,kotlin,cpp,c,cuda,ruby,csharp,visualbasic,dart,php}"
INSTALL_SCIP_JAVA="${INSTALL_SCIP_JAVA:-1}"
SCIP_JAVA_VERSION="${SCIP_JAVA_VERSION:-0.12.3}"
INSTALL_SCIP_DOTNET="${INSTALL_SCIP_DOTNET:-1}"
INSTALL_SCIP_DART="${INSTALL_SCIP_DART:-1}"
INSTALL_SCIP_PHP="${INSTALL_SCIP_PHP:-1}"
INSTALL_SCIP_RUBY="${INSTALL_SCIP_RUBY:-1}"
INSTALL_SCIP_CLANG="${INSTALL_SCIP_CLANG:-1}"
INSTALL_SCIP_CLI="${INSTALL_SCIP_CLI:-1}"

log() {
  printf '\n==> %s\n' "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

path_has() {
  printf '%s' ":$PATH:" | grep -q ":$1:"
}

docker_cmd() {
  if [ "${USE_SUDO_DOCKER:-0}" = "1" ]; then
    sudo docker "$@"
  else
    docker "$@"
  fi
}

sudo_if_needed() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

ensure_package() {
  package="$1"
  command_name="${2:-$1}"

  if have "$command_name"; then
    return
  fi

  case "$(detect_os)" in
    macos)
      have brew || die "Homebrew is required to install $package automatically on macOS. Install it from https://brew.sh and rerun this script."
      brew install "$package"
      ;;
    linux)
      if have apt-get; then
        sudo_if_needed apt-get update
        sudo_if_needed apt-get install -y "$package"
      elif have dnf; then
        sudo_if_needed dnf install -y "$package"
      elif have yum; then
        sudo_if_needed yum install -y "$package"
      elif have pacman; then
        sudo_if_needed pacman -Sy --noconfirm "$package"
      else
        die "Could not install $package automatically. Install it and rerun this script."
      fi
      ;;
  esac
}

detect_os() {
  case "$(uname -s)" in
    Darwin) printf 'macos' ;;
    Linux) printf 'linux' ;;
    *) die "Unsupported OS: $(uname -s)" ;;
  esac
}

ensure_uv() {
  if [ -x "$UV_BIN" ]; then
    return
  fi

  if have uv; then
    UV_BIN="$(command -v uv)"
    return
  fi

  log "Installing uv"
  case "$(detect_os)" in
    macos)
      ensure_package uv uv
      UV_BIN="$(command -v uv)"
      ;;
    linux)
      ensure_package curl curl
      curl -LsSf https://astral.sh/uv/install.sh | sh
      [ -x "$UV_BIN" ] || UV_BIN="$(command -v uv)"
      ;;
  esac

  [ -x "$UV_BIN" ] || have uv || die "uv is still unavailable after installation."
}

find_python() {
  for candidate in "$PYTHON_CMD" python3 python3.14 python3.13 python3.12 python3.11 python3.10 /opt/homebrew/bin/python3 /usr/local/bin/python3; do
    [ -n "$candidate" ] || continue
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
      then
        PYTHON_CMD="$(command -v "$candidate")"
        return
      fi
    fi
  done
}

ensure_python() {
  if find_python; then
    return
  fi

  log "Installing Python 3.10+"
  case "$(detect_os)" in
    macos)
      have brew || die "Homebrew is required to install Python automatically on macOS. Install it from https://brew.sh and rerun this script."
      brew install python
      ;;
    linux)
      if have apt-get; then
        sudo_if_needed apt-get update
        sudo_if_needed apt-get install -y python3 python3-venv python3-pip
      elif have dnf; then
        sudo_if_needed dnf install -y python3 python3-pip
      elif have yum; then
        sudo_if_needed yum install -y python3 python3-pip
      elif have pacman; then
        sudo_if_needed pacman -Sy --noconfirm python python-pip
      else
        die "Could not install Python automatically. Install Python 3.10+ and rerun this script."
      fi
      ;;
  esac

  find_python || die "Python 3.10+ is still unavailable after installation."
}

ensure_docker() {
  if have docker; then
    return
  fi

  log "Installing Docker"
  case "$(detect_os)" in
    macos)
      have brew || die "Homebrew is required to install Docker automatically on macOS. Install it from https://brew.sh and rerun this script."
      brew install --cask docker
      open -a Docker || true
      ;;
    linux)
      if have apt-get; then
        sudo_if_needed apt-get update
        sudo_if_needed apt-get install -y ca-certificates curl gnupg
        curl -fsSL https://get.docker.com | sudo_if_needed sh
      elif have dnf; then
        sudo_if_needed dnf install -y docker
        sudo_if_needed systemctl enable --now docker
      elif have yum; then
        sudo_if_needed yum install -y docker
        sudo_if_needed systemctl enable --now docker
      elif have pacman; then
        sudo_if_needed pacman -Sy --noconfirm docker
        sudo_if_needed systemctl enable --now docker
      else
        die "Could not install Docker automatically. Install Docker and rerun this script."
      fi
      ;;
  esac

  have docker || die "docker is still unavailable after installation."
}

ensure_docker_running() {
  if docker info >/dev/null 2>&1; then
    USE_SUDO_DOCKER=0
    return
  fi

  if have sudo && sudo docker info >/dev/null 2>&1; then
    USE_SUDO_DOCKER=1
    return
  fi

  case "$(detect_os)" in
    macos)
      log "Starting Docker Desktop"
      open -a Docker || true
      for _ in $(seq 1 60); do
        if docker info >/dev/null 2>&1; then
          USE_SUDO_DOCKER=0
          return
        fi
        sleep 2
      done
      die "Docker Desktop did not become ready. Start Docker Desktop manually and rerun this script."
      ;;
    linux)
      if have systemctl; then
        sudo_if_needed systemctl enable --now docker || true
      fi
      for _ in $(seq 1 30); do
        if docker info >/dev/null 2>&1; then
          USE_SUDO_DOCKER=0
          return
        fi
        if have sudo && sudo docker info >/dev/null 2>&1; then
          USE_SUDO_DOCKER=1
          return
        fi
        sleep 2
      done
      die "Docker daemon is not reachable. Start Docker or add your user to the docker group, then rerun this script."
      ;;
  esac
}

install_cgc() {
  log "Installing CodeGraphContext into $CGC_VENV"
  mkdir -p "$CGC_HOME" "$CGC_BIN_DIR"

  ensure_uv
  "$UV_BIN" python install "$CGC_PYTHON_VERSION" >/dev/null

  if [ -x "$CGC_VENV/bin/python" ]; then
    if ! "$CGC_VENV/bin/python" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
    then
      backup="$CGC_VENV.py$(date +%Y%m%d%H%M%S).bak"
      log "Existing CGC venv uses Python < 3.10; moving it to $backup"
      mv "$CGC_VENV" "$backup"
    fi
  fi

  if [ ! -x "$CGC_VENV/bin/python" ]; then
    "$UV_BIN" venv --python "$CGC_PYTHON_VERSION" "$CGC_VENV" >/dev/null
  fi

  "$UV_BIN" pip install --python "$CGC_VENV/bin/python" --upgrade pip setuptools wheel >/dev/null
  "$UV_BIN" pip install --python "$CGC_VENV/bin/python" --upgrade codegraphcontext >/dev/null

  cat > "$CGC_BIN" <<EOF
#!/usr/bin/env bash
export PATH="$CGC_VENV/bin:$CGC_HOME/node/node_modules/.bin:$CGC_BIN_DIR:\$HOME/go/bin:\$HOME/.cargo/bin:\$PATH"
exec "$CGC_VENV/bin/cgc" "\$@"
EOF
  chmod +x "$CGC_BIN"

  if ! printf '%s' ":$PATH:" | grep -q ":$CGC_BIN_DIR:"; then
    log "Add $CGC_BIN_DIR to PATH to use cgc directly from new shells."
  fi
}

ensure_scip_prerequisites() {
  log "Installing SCIP prerequisites"

  case "$(detect_os)" in
    macos)
      ensure_package node npm
      ensure_package go go
      ensure_package rust cargo
      ensure_package ruby ruby
      ensure_package php php
      ensure_package composer composer
      if ! have java; then
        brew install openjdk@17
        export PATH="/opt/homebrew/opt/openjdk@17/bin:/usr/local/opt/openjdk@17/bin:$PATH"
      fi
      if ! have dotnet; then
        brew install --cask dotnet-sdk || brew install dotnet
      fi
      if ! have dart; then
        brew tap dart-lang/dart || true
        brew install dart
      fi
      ;;
    linux)
      ensure_package curl curl
      ensure_package nodejs node
      ensure_package npm npm
      ensure_package golang-go go
      ensure_package cargo cargo
      ensure_package ruby ruby
      ensure_package php-cli php
      ensure_package composer composer
      if ! have java; then
        ensure_package openjdk-17-jre-headless java
      fi
      if ! have dotnet; then
        ensure_package dotnet-sdk-8.0 dotnet
      fi
      if ! have dart; then
        ensure_package dart dart
      fi
      ;;
  esac

  export PATH="$CGC_VENV/bin:$CGC_HOME/node/node_modules/.bin:$CGC_BIN_DIR:$HOME/go/bin:$HOME/.cargo/bin:$HOME/.dotnet/tools:$HOME/.pub-cache/bin:$HOME/.composer/vendor/bin:$HOME/.config/composer/vendor/bin:/opt/homebrew/opt/openjdk@17/bin:/usr/local/opt/openjdk@17/bin:$PATH"
}

install_scip_indexers() {
  log "Installing SCIP indexers"
  ensure_scip_prerequisites

  if have npm; then
    npm install --prefix "$CGC_HOME/node" @sourcegraph/scip-python @sourcegraph/scip-typescript
    export PATH="$CGC_HOME/node/node_modules/.bin:$PATH"
  else
    die "npm is unavailable; cannot install scip-python and scip-typescript."
  fi

  if have go; then
    go install github.com/scip-code/scip-go/cmd/scip-go@latest || go install github.com/sourcegraph/scip-go/cmd/scip-go@latest
  else
    die "go is unavailable; cannot install scip-go."
  fi

  if [ "$INSTALL_SCIP_CLI" = "1" ] && ! have scip; then
    install_scip_cli
  fi

  install_scip_rust_wrapper

  if [ "$INSTALL_SCIP_JAVA" = "1" ] && ! have scip-java; then
    install_scip_java
  fi

  if [ "$INSTALL_SCIP_CLANG" = "1" ] && ! have scip-clang; then
    install_scip_clang
  fi

  if [ "$INSTALL_SCIP_RUBY" = "1" ] && ! have scip-ruby; then
    install_scip_ruby
  fi

  if [ "$INSTALL_SCIP_DOTNET" = "1" ] && ! have scip-dotnet; then
    install_scip_dotnet
  fi

  if [ "$INSTALL_SCIP_DART" = "1" ] && ! have scip-dart; then
    install_scip_dart
  fi

  if [ "$INSTALL_SCIP_PHP" = "1" ] && ! have scip-php; then
    install_scip_php
  fi
}

github_latest_tag() {
  repo="$1"
  python3 - "$repo" <<'PY'
import json
import sys
import urllib.request

repo = sys.argv[1]
with urllib.request.urlopen(f"https://api.github.com/repos/{repo}/releases/latest", timeout=30) as r:
    print(json.load(r)["tag_name"])
PY
}

download_github_release_asset() {
  repo="$1"
  asset="$2"
  output="$3"
  tag="$(github_latest_tag "$repo")"
  curl -fL "https://github.com/$repo/releases/download/$tag/$asset" -o "$output"
}

scip_os_arch() {
  os="$(uname -s)"
  arch="$(uname -m)"
  case "$os:$arch" in
    Darwin:arm64) printf 'darwin arm64 arm64-darwin' ;;
    Darwin:x86_64) printf 'darwin amd64 x86_64-darwin' ;;
    Linux:x86_64) printf 'linux amd64 x86_64-linux' ;;
    Linux:aarch64|Linux:arm64) printf 'linux arm64 arm64-linux' ;;
    *) die "Unsupported OS/arch for SCIP binary downloads: $os/$arch" ;;
  esac
}

install_scip_cli() {
  log "Installing scip CLI"
  mkdir -p "$CGC_BIN_DIR"
  read -r os arch _ < <(scip_os_arch)
  tmp_dir="$(mktemp -d -t cgc-scip.XXXXXX)"
  download_github_release_asset sourcegraph/scip "scip-$os-$arch.tar.gz" "$tmp_dir/scip.tar.gz"
  tar -xzf "$tmp_dir/scip.tar.gz" -C "$tmp_dir"
  found="$(find "$tmp_dir" -type f -name scip -perm -111 | head -1)"
  [ -n "$found" ] || die "Downloaded scip archive did not contain an executable scip binary."
  mv "$found" "$CGC_BIN_DIR/scip"
  chmod +x "$CGC_BIN_DIR/scip"
  rm -rf "$tmp_dir"
}

install_scip_clang() {
  log "Installing scip-clang"
  mkdir -p "$CGC_BIN_DIR"
  case "$(uname -s):$(uname -m)" in
    Darwin:arm64) asset="scip-clang-arm64-darwin" ;;
    Linux:x86_64) asset="scip-clang-x86_64-linux" ;;
    *) die "No prebuilt scip-clang binary is available for $(uname -s)/$(uname -m)." ;;
  esac
  download_github_release_asset sourcegraph/scip-clang "$asset" "$CGC_BIN_DIR/scip-clang"
  chmod +x "$CGC_BIN_DIR/scip-clang"
}

install_scip_ruby() {
  log "Installing scip-ruby"
  mkdir -p "$CGC_BIN_DIR"
  case "$(uname -s):$(uname -m)" in
    Darwin:arm64) asset="scip-ruby-arm64-darwin" ;;
    Linux:x86_64) asset="scip-ruby-x86_64-linux" ;;
    *) die "No prebuilt scip-ruby binary is available for $(uname -s)/$(uname -m)." ;;
  esac
  download_github_release_asset sourcegraph/scip-ruby "$asset" "$CGC_BIN_DIR/scip-ruby"
  chmod +x "$CGC_BIN_DIR/scip-ruby"
}

install_scip_dotnet() {
  log "Installing scip-dotnet"
  have dotnet || die "dotnet is unavailable; cannot install scip-dotnet."
  dotnet tool install --global scip-dotnet || dotnet tool update --global scip-dotnet
  export PATH="$HOME/.dotnet/tools:$PATH"
}

install_scip_dart() {
  log "Installing scip-dart"
  have dart || die "dart is unavailable; cannot install scip-dart."
  dart pub global activate scip_dart
  mkdir -p "$CGC_BIN_DIR"
  cat > "$CGC_BIN_DIR/scip-dart" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exec dart pub global run scip_dart "$@"
EOF
  chmod +x "$CGC_BIN_DIR/scip-dart"
}

install_scip_php() {
  log "Installing scip-php"
  have composer || die "composer is unavailable; cannot install scip-php."
  composer global require davidrjenni/scip-php
  mkdir -p "$CGC_BIN_DIR"
  cat > "$CGC_BIN_DIR/scip-php" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
for bin in "$HOME/.composer/vendor/bin/scip-php" "$HOME/.config/composer/vendor/bin/scip-php"; do
  if [ -x "$bin" ]; then
    exec "$bin" "$@"
  fi
done
echo "scip-php composer global binary not found" >&2
exit 1
EOF
  chmod +x "$CGC_BIN_DIR/scip-php"
}

install_scip_rust_wrapper() {
  log "Installing scip-rust wrapper"

  if have rustup; then
    rustup component add rust-analyzer
  elif ! have rust-analyzer; then
    ensure_package rust-analyzer rust-analyzer
  fi

  have rust-analyzer || die "rust-analyzer is unavailable; cannot install scip-rust wrapper."

  mkdir -p "$CGC_BIN_DIR"
  cat > "$CGC_BIN_DIR/scip-rust" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

output="index.scip"
path="."

while [ "$#" -gt 0 ]; do
  case "$1" in
    index)
      shift
      ;;
    --output)
      output="$2"
      shift 2
      ;;
    --output=*)
      output="${1#--output=}"
      shift
      ;;
    --help|-h)
      rust-analyzer --help
      exit 0
      ;;
    *)
      path="$1"
      shift
      ;;
  esac
done

rust-analyzer scip "$path" --output "$output"
EOF
  chmod +x "$CGC_BIN_DIR/scip-rust"
}

install_scip_java() {
  log "Installing scip-java"

  have java || die "java is unavailable; cannot install scip-java."
  ensure_package curl curl

  mkdir -p "$CGC_BIN_DIR"
  coursier_tmp="$(mktemp -t cgc-coursier.XXXXXX)"
  curl -fLo "$coursier_tmp" https://git.io/coursier-cli
  chmod +x "$coursier_tmp"
  "$coursier_tmp" bootstrap \
    --standalone \
    -o "$CGC_BIN_DIR/scip-java" \
    "com.sourcegraph:scip-java_2.13:$SCIP_JAVA_VERSION" \
    --main com.sourcegraph.scip_java.ScipJava
  rm -f "$coursier_tmp"
  chmod +x "$CGC_BIN_DIR/scip-java"
}

start_neo4j() {
  log "Starting local Neo4j in Docker"

  docker_cmd pull "$NEO4J_IMAGE"

  if docker_cmd ps -a --format '{{.Names}}' | grep -qx "$NEO4J_CONTAINER_NAME"; then
    current_image="$(docker_cmd inspect -f '{{.Config.Image}}' "$NEO4J_CONTAINER_NAME")"
    if [ "$current_image" != "$NEO4J_IMAGE" ]; then
      log "Existing container $NEO4J_CONTAINER_NAME uses $current_image; leaving it in place."
    fi
    docker_cmd start "$NEO4J_CONTAINER_NAME" >/dev/null
  else
    docker_cmd volume create cgc_neo4j_data >/dev/null
    docker_cmd volume create cgc_neo4j_logs >/dev/null
    docker_cmd run -d \
      --name "$NEO4J_CONTAINER_NAME" \
      -p "$NEO4J_HTTP_PORT:7474" \
      -p "$NEO4J_BOLT_PORT:7687" \
      -e "NEO4J_AUTH=$NEO4J_USER/$NEO4J_PASSWORD" \
      -e "NEO4J_server_memory_heap_initial__size=512m" \
      -e "NEO4J_server_memory_heap_max__size=2G" \
      -v cgc_neo4j_data:/data \
      -v cgc_neo4j_logs:/logs \
      "$NEO4J_IMAGE" >/dev/null
  fi
}

wait_for_neo4j() {
  log "Waiting for Neo4j Bolt endpoint"
  for _ in $(seq 1 90); do
    if docker_cmd exec "$NEO4J_CONTAINER_NAME" cypher-shell \
      -u "$NEO4J_USER" \
      -p "$NEO4J_PASSWORD" \
      'RETURN 1;' >/dev/null 2>&1; then
      return
    fi
    sleep 2
  done

  docker_cmd logs --tail 80 "$NEO4J_CONTAINER_NAME" >&2 || true
  die "Neo4j did not become ready."
}

set_cgc_env() {
  key="$1"
  value="$2"
  env_file="$CGC_HOME/.env"
  tmp_file="$env_file.tmp"

  mkdir -p "$CGC_HOME"
  touch "$env_file"

  if grep -q "^$key=" "$env_file"; then
    while IFS= read -r line || [ -n "$line" ]; do
      case "$line" in
        "$key="*) printf '%s=%s\n' "$key" "$value" ;;
        *) printf '%s\n' "$line" ;;
      esac
    done < "$env_file" > "$tmp_file"
    mv "$tmp_file" "$env_file"
  else
    printf '%s=%s\n' "$key" "$value" >> "$env_file"
  fi
}

configure_cgc() {
  log "Configuring CodeGraphContext to use local Neo4j and SCIP"
  mkdir -p "$CGC_HOME"

  set_cgc_env DEFAULT_DATABASE neo4j
  set_cgc_env CGC_GRAPH_BACKEND neo4j
  set_cgc_env NEO4J_URI "$NEO4J_URI"
  set_cgc_env NEO4J_USERNAME "$NEO4J_USER"
  set_cgc_env NEO4J_USER "$NEO4J_USER"
  set_cgc_env NEO4J_PASSWORD "$NEO4J_PASSWORD"
  set_cgc_env NEO4J_DATABASE "$NEO4J_DATABASE"
  set_cgc_env SCIP_INDEXER true
  set_cgc_env SCIP_LANGUAGES "$SCIP_LANGUAGES"

  if "$CGC_BIN" config --help >/dev/null 2>&1; then
    "$CGC_BIN" config db neo4j >/dev/null
    "$CGC_BIN" config set SCIP_INDEXER true >/dev/null
    "$CGC_BIN" config set SCIP_LANGUAGES "$SCIP_LANGUAGES" >/dev/null
  elif "$CGC_BIN" default --help >/dev/null 2>&1; then
    "$CGC_BIN" default database neo4j >/dev/null
  fi
}

verify_installation() {
  log "Verifying installation"
  "$CGC_BIN" --help >/dev/null 2>&1
  docker_cmd ps --filter "name=^/${NEO4J_CONTAINER_NAME}$" --format 'Neo4j container: {{.Names}} {{.Status}}'
  "$CGC_BIN" --version
  printf 'CGC config: %s\n' "$CGC_HOME/.env"
  printf 'SCIP tools:\n'
  for tool in scip scip-python scip-typescript scip-go scip-rust scip-java scip-clang scip-ruby scip-dotnet scip-dart scip-php; do
    if command -v "$tool" >/dev/null 2>&1; then
      printf '  %-16s %s\n' "$tool" "$(command -v "$tool")"
    else
      printf '  %-16s %s\n' "$tool" "not found"
    fi
  done
}

main() {
  ensure_python
  ensure_docker
  ensure_docker_running
  install_cgc
  install_scip_indexers
  start_neo4j
  wait_for_neo4j
  configure_cgc
  verify_installation

  cat <<EOF

CodeGraphContext is ready.

CLI:
  $CGC_BIN --help
  $CGC_BIN index /path/to/repo

Neo4j:
  Browser: http://localhost:$NEO4J_HTTP_PORT
  Bolt:    $NEO4J_URI
  User:    $NEO4J_USER
  Pass:    $NEO4J_PASSWORD

If cgc is not on PATH yet, either run it as:
  $CGC_BIN

or add this to your shell profile:
  export PATH="$CGC_BIN_DIR:\$PATH"
EOF
}

main "$@"
