FROM python:3.14-trixie

RUN \
    # 1. Define the file path for convenience
    FILE=/etc/apt/sources.list.d/debian.sources && \
    # 2. Backup the original file
    cp "$FILE" "$FILE.bak" && \
    # 3. Replace default URLs with Aliyun mirror
    #    Trixie uses 'deb.debian.org' and 'security.debian.org' inside this file
    sed -i 's@deb.debian.org@mirrors.aliyun.com@g' "$FILE" && \
    sed -i 's@security.debian.org@mirrors.aliyun.com@g' "$FILE" && \
    # 4. Install your packages
    apt-get update && apt-get install -y \
    bash \
    git \
    curl \
    jq && \
    # 5. Restore the original file (send it back)
    mv "$FILE.bak" "$FILE" && \
    # 6. Clear the cache so the final image doesn't contain mirror metadata
    rm -rf /var/lib/apt/lists/*

# Set workspace as working directory
WORKDIR /workspace

# Volume for persistent workspace
VOLUME ["/workspace"]

# Run bash interactively to keep container alive
CMD ["bash"]
